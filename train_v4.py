
import json
import torch

from model_v4 import MiniGPTV4, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

batch_size = 32
max_iters = 3000
eval_interval = 200
eval_iters = 30
learning_rate = 3e-4

patience = 5
min_delta = 0.001

# 2. Load vocabulary
with open("data/vocab_v4.json", "r", encoding="utf-8") as f:
    chars = json.load(f)["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

# 3. Load datasets
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    return torch.tensor(
        encode(text),
        dtype=torch.long,
        device=device
    )

train_data = load_data("data/train_v4.txt")
val_data = load_data("data/val_v4.txt")

print("Device:", device)
print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))

# 4. Create random batches
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
model = MiniGPTV4().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

# 6. Evaluate
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

# 7. Training
history = []

best_val_loss = float("inf")
best_step = 0
no_improvement = 0

print("\n--- MiniGPT V4 Training ---")

for step in range(max_iters + 1):

    if step % eval_interval == 0:
        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        history.append({
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(
            f"Step {step:4d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}",
            flush=True
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_step = step
            no_improvement = 0

            torch.save(
                model.state_dict(),
                "mini_gpt_v4_best.pth"
            )

            print("  Best checkpoint saved!", flush=True)

        else:
            no_improvement += 1

            print(
                f"  No improvement: "
                f"{no_improvement}/{patience}",
                flush=True
            )

        if no_improvement >= patience:
            print("\nEarly stopping triggered!")
            break

    if step == max_iters:
        break

    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 8. Save training history
with open("data/loss_history_v4.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n--- Training Finished ---")
print("Best step:", best_step)
print("Best validation loss:", round(best_val_loss, 4))
print("Model saved to mini_gpt_v4_best.pth")
print("History saved to data/loss_history_v4.json")
