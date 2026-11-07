
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F

from model_v4 import MiniGPTV4, block_size

# Configuration
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

# Load trained model
model = MiniGPTV4().to(device)

model.load_state_dict(
    torch.load(
        "mini_gpt_v4_best.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

@torch.no_grad()
def generate(
    start_text,
    max_new_tokens=300,
    temperature=0.8,
    top_k=20,
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

        # Top-k filtering
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            threshold = values[:, [-1]]
            logits = logits.masked_fill(
                logits < threshold, float("-inf")
            )

        # Top-p filtering
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(
            sorted_probs, dim=-1
        )

        remove = cumulative_probs - sorted_probs >= top_p

        sorted_logits = sorted_logits.masked_fill(
            remove, float("-inf")
        )

        probs = F.softmax(sorted_logits, dim=-1)

        next_sorted = torch.multinomial(
            probs, num_samples=1
        )

        idx_next = sorted_indices.gather(
            -1, next_sorted
        )

        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())



# Interactive interface
print("\n================================")
print("     MiniGPT V4 Generator")
print("================================")
print("Device:", device)
print("Type 'exit' to quit.")
print()

while True:
    prompt = input("Enter your prompt: ").strip()

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    if not prompt:
        continue

    unknown = [c for c in prompt if c not in stoi]

    if unknown:
        print("Unsupported characters:", unknown)
        continue

    try:
        temp_input = input("Temperature [0.8]: ").strip()
        temperature = float(temp_input) if temp_input else 0.8

        k_input = input("Top-k [20]: ").strip()
        top_k = int(k_input) if k_input else 20

        p_input = input("Top-p [0.9]: ").strip()
        top_p = float(p_input) if p_input else 0.9

        length_input = input("New tokens [300]: ").strip()
        max_new_tokens = int(length_input) if length_input else 300

        if not (temperature > 0):
            raise ValueError("Temperature must be positive.")
        if not (top_k >= 1):
            raise ValueError("Top-k must be at least 1.")
        if not (0 < top_p <= 1):
            raise ValueError("Top-p must be between 0 and 1.")
        if not (max_new_tokens >= 1):
            raise ValueError("New tokens must be at least 1.")

    except ValueError as e:
        print("Invalid parameter:", e)
        continue

    print("\nGenerating...\n")

    result = generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    print(result)
    
    # Save generated text
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = output_dir / f"generation_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("MiniGPT V4 Generation\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Top-k: {top_k}\n")
        f.write(f"Top-p: {top_p}\n")
        f.write(f"New tokens: {max_new_tokens}\n")
        f.write("\nGenerated Text:\n\n")
        f.write(result)

    print(f"\nSaved to: {output_file}")

    print("\n" + "-" * 50 + "\n")

