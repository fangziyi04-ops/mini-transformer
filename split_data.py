
import random

random.seed(42)

# Read original sentences, not repeated V3 data
with open("input.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

random.shuffle(lines)

# 80% training, 20% validation
split_idx = int(0.8 * len(lines))

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

# Save datasets
with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines) + "\n")

with open("val.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val_lines) + "\n")

print("--- Dataset Split ---")
print("Total sentences:", len(lines))
print("Training sentences:", len(train_lines))
print("Validation sentences:", len(val_lines))
print("Train characters:", len(open("train.txt", encoding="utf-8").read()))
print("Validation characters:", len(open("val.txt", encoding="utf-8").read()))
print("Saved train.txt and val.txt")
