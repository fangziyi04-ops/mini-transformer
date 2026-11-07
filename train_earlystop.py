
import json
import torch

from model import MiniGPT, block_size

# 1. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

batch_size = 16
max_iters = 2000
eval_interval = 50

patience = 4
min_delta = 0.01
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


# 7. Training with Early Stopping

history = []

best_val_loss = float("inf")
best_step = 0
no_improvement = 0

print("\n--- Training with Early Stopping ---")

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
            f"Val Loss: {val_loss:.4f}"
        )

        # Save model when validation improves
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_step = step
            no_improvement = 0

            torch.save(
                model.state_dict(),
                "mini_gpt_best.pth"
            )

            print("  New best model saved!")

        else:
            no_improvement += 1

            print(
                f"  No improvement: "
                f"{no_improvement}/{patience}"
            )

        # Stop training if no improvement
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

# 8. Save experiment history

with open("earlystop_history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n--- Training Finished ---")
print("Best step:", best_step)
print("Best validation loss:", round(best_val_loss, 4))
print("Best model saved to mini_gpt_best.pth")
print("History saved to earlystop_history.json")
