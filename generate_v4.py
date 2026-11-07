
import json
import torch
import torch.nn.functional as F

from model_v4 import MiniGPTV4, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# 2. Load V4 vocabulary
with open("data/vocab_v4.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[c] for c in text]

def decode(ids):
    return "".join(itos[i] for i in ids)

# 3. Load trained model
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
print("V4 model loaded successfully!")

# 4. Generate text
@torch.no_grad()
def generate(start_text, max_new_tokens=500, temperature=0.8):

    idx = torch.tensor(
        [encode(start_text)],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

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


# 5. Test generation
print("\n--- MiniGPT V4 Generated Text ---")

print(
    generate(
        "Sherlock Holmes",
        max_new_tokens=500,
        temperature=0.8
    )
)
