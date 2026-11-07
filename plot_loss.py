
import json
import matplotlib.pyplot as plt

# 1. Load training history
with open("loss_history.json", "r") as f:
    history = json.load(f)

# 2. Extract data
steps = [item["step"] for item in history]
train_losses = [item["train_loss"] for item in history]
val_losses = [item["val_loss"] for item in history]

# 3. Find the best validation loss
best_idx = val_losses.index(min(val_losses))
best_step = steps[best_idx]
best_val_loss = val_losses[best_idx]

# 4. Plot loss curves
plt.figure(figsize=(9, 5))

plt.plot(
    steps, train_losses,
    label="Training Loss",
    linewidth=2
)

plt.plot(
    steps, val_losses,
    label="Validation Loss",
    linewidth=2
)

# Mark the best validation point
plt.scatter(
    best_step, best_val_loss,
    color="red",
    s=70,
    zorder=5,
    label=f"Best Validation (Step {best_step})"
)

plt.xlabel("Training Steps")
plt.ylabel("Cross-Entropy Loss")
plt.title("MiniGPT - Training vs Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 5. Save figure
plt.savefig("loss_curve.png", dpi=300)

print("--- Loss Analysis ---")
print("Best validation step:", best_step)
print("Best validation loss:", round(best_val_loss, 4))
print("Final training loss:", round(train_losses[-1], 4))
print("Final validation loss:", round(val_losses[-1], 4))
print("Figure saved to loss_curve.png")

plt.show()
