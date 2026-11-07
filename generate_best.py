
import json
import torch
import torch.nn.functional as F

from model import MiniGPT, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load vocabulary
with open("vocab.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

# 3. Load best checkpoint
model = MiniGPT().to(device)

model.load_state_dict(
    torch.load(
        "mini_gpt_best.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Device:", device)
print("Best checkpoint loaded successfully!")

# 4. Autoregressive generation
@torch.no_grad()
def generate(start_text, max_new_tokens=200):
    idx = torch.tensor(
        [encode(start_text)],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())


# 5. Generate text
print("\n--- Best Model Generated Text ---")
print(generate("Transformers", 200))
