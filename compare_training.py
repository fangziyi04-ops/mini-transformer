
import json
import matplotlib.pyplot as plt

# Load ordinary training history
with open("loss_history.json", "r") as f:
    normal = json.load(f)

# Load early stopping history
with open("earlystop_history.json", "r") as f:
    early = json.load(f)

normal_steps = [x["step"] for x in normal]
normal_val = [x["val_loss"] for x in normal]

early_steps = [x["step"] for x in early]
early_val = [x["val_loss"] for x in early]

# Find best checkpoint
best_idx = early_val.index(min(early_val))
best_step = early_steps[best_idx]
best_loss = early_val[best_idx]

plt.figure(figsize=(9, 5))

plt.plot(
    normal_steps,
    normal_val,
    label="Normal Training - Validation Loss",
    linewidth=2
)

plt.plot(
    early_steps,
    early_val,
    label="Early Stopping - Validation Loss",
    linewidth=2,
    marker="o"
)

plt.scatter(
    best_step,
    best_loss,
    color="red",
    s=90,
    zorder=5,
    label="Best Checkpoint"
)

plt.axvline(
    best_step,
    linestyle="--",
    color="gray",
    alpha=0.6
)

plt.xlabel("Training Steps")
plt.ylabel("Validation Loss")
plt.title("MiniGPT - Normal Training vs Early Stopping")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("training_comparison.png", dpi=300)

print("--- Training Comparison ---")
print("Normal final validation loss:", round(normal_val[-1], 4))
print("Early stopping best step:", best_step)
print("Early stopping best validation loss:", round(best_loss, 4))
print("Saved to training_comparison.png")

plt.show()
