
import json
import torch
import torch.nn.functional as F

from model_v4 import MiniGPTV4, block_size

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocabulary
with open("data/vocab_v4.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

# Load model
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

@torch.no_grad()
def generate(
    start_text,
    max_new_tokens=300,
    temperature=0.8,
    top_p=0.9
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

        # Temperature
        logits = logits / temperature

        # Sort probabilities
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)

        cumulative_probs = torch.cumsum(
            sorted_probs, dim=-1
        )

        # Remove tokens beyond top-p
        sorted_remove = cumulative_probs - sorted_probs >= top_p

        sorted_logits = sorted_logits.masked_fill(
            sorted_remove,
            float("-inf")
        )

        # Sample from retained candidates
        probs = F.softmax(sorted_logits, dim=-1)

        next_sorted = torch.multinomial(
            probs, num_samples=1
        )

        idx_next = sorted_indices.gather(
            -1, next_sorted
        )

        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())


# Compare top-p
top_p_values = [0.5, 0.7, 0.9, 1.0]

print("\n--- Top-p Sampling Comparison ---")

for p in top_p_values:

    torch.manual_seed(42)

    print(f"\n--- Top-p = {p} ---")

    result = generate(
        "Sherlock Holmes",
        max_new_tokens=300,
        temperature=0.8,
        top_p=p
    )

    print(result)
    print("-" * 50)
