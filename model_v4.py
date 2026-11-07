
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Model configuration
with open("data/vocab_v4.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

vocab_size = len(chars)

block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1


# 2. Single causal attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v
        return out


# 3. Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.proj(out)
        out = self.dropout(out)

        return out


# 4. Feed-forward network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 5. Transformer block
class Block(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# 6. Complete MiniGPT V4
class MiniGPTV4(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss


# 7. Model test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniGPTV4().to(device)

    x = torch.randint(
        0,
        vocab_size,
        (2, block_size),
        device=device
    )

    logits, loss = model(x, x)

    print("--- MiniGPT V4 Model Test ---")
    print("Device:", device)
    print("Vocabulary size:", vocab_size)
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print("Loss:", round(loss.item(), 4))
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )
    print("Model test passed!")
