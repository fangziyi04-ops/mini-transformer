
# MiniGPT: Transformer Language Model from Scratch

A character-level, decoder-only Transformer language model implemented from scratch using PyTorch.

This project explores the core architecture of GPT-style language models, including causal self-attention, multi-head attention, positional embeddings, autoregressive text generation, and different sampling strategies.

The model was trained on *The Adventures of Sherlock Holmes* by Arthur Conan Doyle.

## 1. Project Overview

The goal of this project is to understand how Transformer-based language models work by implementing and training a small GPT-style model.

The project covers the complete machine learning workflow:

- Text preprocessing
- Character-level tokenization
- Transformer architecture implementation
- Training and validation
- Early stopping and model checkpointing
- Autoregressive text generation
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Interactive text generation

## 2. Model Architecture

The final model (MiniGPT V4) uses a decoder-only Transformer architecture.

| Component | Configuration |
|---|---|
| Model type | Decoder-only Transformer |
| Tokenization | Character-level |
| Vocabulary size | 88 |
| Context length | 128 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.1 |
| Parameters | 830,808 |
| Framework | PyTorch |

### Architecture

```text
Input Characters
       |
       v
Character Tokenization
       |
       v
Token Embeddings + Positional Embeddings
       |
       v
4 Transformer Blocks
       |
       +-- Multi-Head Causal Self-Attention
       |
       +-- Feed-Forward Network
       |
       v
Final Layer Normalization
       |
       v
Linear Language Modeling Head
       |
       v
Next-Character Prediction
```

## 3. Dataset

The training dataset is *The Adventures of Sherlock Holmes* by Arthur Conan Doyle, obtained from Project Gutenberg.

Dataset preprocessing includes:

- Removing Project Gutenberg headers and footers
- Normalizing whitespace
- Building a character-level vocabulary
- Splitting the dataset into training and validation sets

Dataset statistics:

| Metric | Value |
|---|---|
| Cleaned text | 562,138 characters |
| Training text | 505,924 characters |
| Validation text | 56,214 characters |
| Vocabulary | 88 characters |
| Train/validation split | 90% / 10% |

The full training text is not included in this repository.

Dataset source:
https://www.gutenberg.org/ebooks/1661

## 4. Training

The model was trained using next-character prediction with cross-entropy loss.

### Training configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 0.0003 |
| Optimizer | AdamW |
| Training iterations | 3,000 |
| Evaluation interval | 200 |
| Evaluation batches | 30 |
| Early stopping patience | 5 |

Training was performed using an NVIDIA GPU with CUDA acceleration.

### Results

At iteration 3,000:

- Training loss: approximately 1.614
- Validation loss: approximately 1.642

The training and validation losses decreased during training, without a clear sign of overfitting in the final observed results.

## 5. Text Generation

The trained model generates text autoregressively.

At each generation step:

1. Encode the input characters.
2. Pass the current context through the Transformer.
3. Predict the probability distribution for the next character.
4. Apply the selected sampling strategy.
5. Sample the next character.
6. Append the sampled character to the sequence.
7. Repeat until the requested length is reached.

### Sampling strategies

**Temperature sampling**

Controls the sharpness of the probability distribution.

**Top-k sampling**

Restricts sampling to the k highest-probability characters.

**Top-p sampling**

Selects the smallest set of high-probability characters whose cumulative probability reaches a specified threshold.

These methods can be combined to control the diversity of generated text.

## 6. Interactive Generator

The project includes a terminal-based interactive generator.

Run:

```bash
python chat_v4.py
```

Example:

```text
Enter your prompt: Sherlock Holmes
Temperature [0.8]: 0.7
Top-k [20]: 10
Top-p [0.9]: 0.9
New tokens [300]: 500
```

Generated text and configuration parameters can be automatically saved in the `outputs/` directory.

## 7. Project Structure

```text
mini-transformer/
├── model_v4.py
├── train_v4.py
├── generate_v4.py
├── chat_v4.py
├── compare_temperature.py
├── compare_topk.py
├── compare_topp.py
├── clean_data.py
├── tokenizer_v4.py
├── split_data_v4.py
├── data/
│   └── vocab_v4.json
├── outputs/
├── .gitignore
└── README.md
```

The repository also contains earlier experiments and model versions developed during the learning process.

## 8. Requirements

- Python 3.12
- PyTorch
- CUDA-compatible GPU (optional)

Install PyTorch following the instructions at:

https://pytorch.org/get-started/locally/

## 9. How to Run

### Step 1: Obtain the dataset

Download *The Adventures of Sherlock Holmes* from Project Gutenberg and save it as:

```text
data/sherlock.txt
```

### Step 2: Preprocess the dataset

```bash
python clean_data.py
python tokenizer_v4.py
python split_data_v4.py
```

### Step 3: Train the model

```bash
python train_v4.py
```

Training saves the best model checkpoint as:

```text
mini_gpt_v4_best.pth
```

### Step 4: Generate text

```bash
python generate_v4.py
```

Or start the interactive generator:

```bash
python chat_v4.py
```

### Step 5: Compare sampling strategies

```bash
python compare_temperature.py
python compare_topk.py
python compare_topp.py
```

## 10. Limitations

This is an educational language model rather than a production-ready chatbot.

Its main limitations include:

- Small model size
- Limited training data
- Character-level tokenization
- Short context window
- Frequent spelling and grammatical errors
- Limited long-range semantic coherence

The model demonstrates the fundamental mechanics of autoregressive Transformer language modeling but does not achieve the language quality of large pretrained models.

## 11. Future Improvements

Potential future directions include:

- Subword tokenization using BPE
- Larger training datasets
- Improved training schedules
- Learning rate scheduling
- More efficient attention implementations
- Quantitative evaluation of generation quality
- A web-based text generation interface

## 12. Key Learning Outcomes

Through this project, I implemented and experimented with:

- Transformer neural network architecture
- Scaled dot-product attention
- Causal masking
- Multi-head self-attention
- Residual connections and layer normalization
- Language model training and evaluation
- Overfitting detection and early stopping
- GPU-accelerated training
- Autoregressive decoding
- Probabilistic sampling strategies

The project provides hands-on experience with the foundations of modern GPT-style language models.

## Acknowledgments

Dataset: *The Adventures of Sherlock Holmes* by Arthur Conan Doyle, via Project Gutenberg.

Framework: PyTorch.
