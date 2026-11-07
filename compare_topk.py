
import json
import torch
import torch.nn.functional as F

from model_v4 import MiniGPTV4, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/vocab_v4.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

# 2. Load trained model
model = MiniGPTV4().to(device)

model.load_state_dict(
    torch.load(
        "mini_gpt_v4_best.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Device:", device)
print("Model loaded successfully!")

# 3. Top-k generation
@torch.no_grad()
def generate(
    start_text,
    max_new_tokens=300,
    temperature=0.8,
    top_k=10
):
    idx = torch.tensor(
        [encode(start_text)],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Temperature scaling
        logits = logits / temperature

        # Keep only top-k logits
        if top_k is not None:
            k = min(top_k, logits.size(-1))

            top_values, _ = torch.topk(
                logits,
                k
            )

            threshold = top_values[:, [-1]]

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf")
            )

        # Sampling
        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

    return decode(idx[0].tolist())


# 4. Compare different top-k values
top_k_values = [5, 10, 20, None]

print("\n--- Top-k Sampling Comparison ---")

for k in top_k_values:

    torch.manual_seed(42)

    print(f"\n--- Top-k = {k} ---")

    text = generate(
        "Sherlock Holmes",
        max_new_tokens=300,
        temperature=0.8,
        top_k=k
    )

    print(text)
    print("-" * 50)
