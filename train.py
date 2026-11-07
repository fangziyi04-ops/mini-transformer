
import json
import torch

from model import MiniGPT, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
max_iters = 3000
learning_rate = 1e-3

torch.manual_seed(42)

# 2. Load training text
with open("input_v3.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. Load vocabulary
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

chars = vocab["chars"]
stoi = {ch: i for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

# 4. Encode the entire dataset
data = torch.tensor(
    encode(text),
    dtype=torch.long,
    device=device
)

print("Device:", device)
print("Total characters:", len(data))
print("Vocabulary size:", len(chars))

# 5. Create random training batches
def get_batch():
    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
        device=device
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x, y

# 6. Initialize model
model = MiniGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)

# 7. Training
print("\n--- MiniGPT V2 Training Started ---")

model.train()

for step in range(max_iters):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 200 == 0 or step == max_iters - 1:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")

# 8. Save trained model
torch.save(
    model.state_dict(),
    "mini_gpt_v3.pth"
)

print("\n--- Training Finished ---")
print("Model saved to mini_gpt_v3.pth")
