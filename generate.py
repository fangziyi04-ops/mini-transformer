import json
import torch
import torch.nn.functional as F

from model import MiniGPT, block_size

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the vocabulary used during training
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

chars = vocab["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]

def decode(numbers):
    return "".join(itos[i] for i in numbers)


# Load trained model
model = MiniGPT().to(device)

model.load_state_dict(
    torch.load(
        "mini_gpt_v3.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Device:", device)
print("Model loaded successfully!")


@torch.no_grad()
def generate(start_text, max_new_tokens=100):
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

        idx_next = torch.multinomial(
            probs, num_samples=1
        )

        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())


print("\n--- Generated Text ---")
print(generate("Transformers are", 300))
