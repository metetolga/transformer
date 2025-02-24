# An explanation is all we need

In the realm of natural language processing, transformer architecture from 2017 always look intimidating to me. In this repository, I've been trying to explore transformer architecture by making it from scratch and training it on a real dataset.

## Contents

* Sequential models
* Attention with Sequential models
* Transformer

I have thought seeing the sequential models would be beneficial for understanding of transformer architecture.  

## Sequential Models

Sequential models process data in a sequential manner, meaning they handle one element at a time, maintaining dependencies across time steps. They are commonly used for time series, natural language processing (NLP), and other sequence-based tasks.

Some examples of Sequential models **RNN, LSTM and GRU**:

* Captures sequential dependencies in data.
* Effective for short-term context retention.
* Can model variable-length input sequences

**But**:

* Due to repeated multiplications of gradients across time steps, small values shrink exponentially (vanishing gradient), while large values grow uncontrollably (exploding gradient). This worsens with longer sequences.  
* Information from earlier time steps is gradually overwritten by newer updates, making it difficult to retain long-term dependencies. Gated architectures like LSTMs and GRUs mitigate this issue but struggle with very long sequences.
* Cannot be parallelized effectively, leading to inefficient training on large models. Since RNNs process inputs sequentially, each step depends on the previous one, preventing parallel execution.

## Attention with Sequential Models

Sequential models with attention aim to overcome the limitations of traditional RNNs, particularly in handling long-range dependencies and parallelization.  
Instead of relying solely on the last hidden state, attention allows the model to dynamically “attend” to relevant parts of the input sequence at each step. More specifically, in classical recurrent networks have trust in last hidden state, yet this could be misleading. With attention input to decoder step is weighted average of every hidden state in encoder.  
Thus, attention provides direct access to all past information, making it easier to capture long-term dependencies.

## Transformer  

[[WORK IN PROGRESS]]
