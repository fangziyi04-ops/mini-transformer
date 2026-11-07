
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load vocabulary
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Model configuration
vocab_size = len(vocab["chars"])
block_size = 64
n_embd = 32
num_heads = 4



class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_size = head_size

    def forward(self, x):
        T = x.shape[-2]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores * (self.head_size ** -0.5)

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        assert n_embd % num_heads == 0

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(
            n_embd, num_heads, block_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            TransformerBlock(n_embd, num_heads, block_size),
            TransformerBlock(n_embd, num_heads, block_size)
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat, targets_flat
            )

        return logits, loss
