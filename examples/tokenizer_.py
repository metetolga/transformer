from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

sequence = 'using a tokenizer from huggingface'
tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

decoded = tokenizer.decode(ids)
print(decoded)
