from model import build_transformer
from dataset import TranslationDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import logging

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def validstep(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_trg = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_ds = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_tgt'], config['seq'])
    val_ds = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_tgt'], config['seq'])

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_trg.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg


def trainstep(model:nn.Module, batch_iterator, optimizer, criterion, device, trg_vocab_size, writer, global_step):
    model.train()
    model = model.to(device)
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device) 
        decoder_mask = batch['decoder_mask'].to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output) 

        label = batch['label'].to(device) 

        loss = criterion(proj_output.view(-1, trg_vocab_size), label.view(-1))
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        writer.add_scalar('train loss', loss.item(), global_step)
        writer.flush()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

def main():
    logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.DEBUG)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'using device: {device}')

    config = get_config()
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_dataset(config)

    src_vocab_len, trg_vocab_len = tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()
    logging.debug(f'source lang length {src_vocab_len} | target lang length {trg_vocab_len}')

    # for quick train there is just 1 encoder and decoder layer
    model = build_transformer(src_vocab_len, trg_vocab_len, config["seq"], config['seq'], d_model=config['d_model'], N=1)
    logging.debug('model is created')

    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
    logging.debug('optimizer and criterion created')
    
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    
    # making non-continous training  
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')

        trainstep(model, batch_iterator, optimizer, criterion, device, trg_vocab_len, writer, global_step)
        validstep(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")    
        torch.save({'epoch': epoch, 
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'global_step': global_step}, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
