
import json
import torch

from model import MiniGPT, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

batch_size = 16
max_iters = 1000
eval_interval = 100
eval_iters = 20
learning_rate = 1e-3

# 2. Load vocabulary
with open("vocab.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

# 3. Load separate datasets
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    return torch.tensor(
        encode(text),
        dtype=torch.long,
        device=device
    )

train_data = load_data("train.txt")
val_data = load_data("val.txt")

print("Device:", device)
print("Train characters:", len(train_data))
print("Validation characters:", len(val_data))

# 4. Random batches
def get_batch(split):
    data = train_data if split == "train" else val_data

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

# 5. Initialize model
model = MiniGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# 6. Evaluate without updating parameters
@torch.no_grad()
def estimate_loss():
    results = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()
    return results

# 7. Training and evaluation
history = []

print("\n--- Training with Validation ---")

for step in range(max_iters + 1):

    if step % eval_interval == 0:
        losses = estimate_loss()

        history.append({
            "step": step,
            "train_loss": losses["train"],
            "val_loss": losses["val"]
        })

        print(
            f"Step {step:4d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )

    if step == max_iters:
        break

    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 8. Save evaluation history
with open("loss_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n--- Experiment Finished ---")
print("Saved loss_history.json")
