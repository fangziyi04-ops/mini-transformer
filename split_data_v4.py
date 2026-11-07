
from pathlib import Path

# 1. Load cleaned text
text_path = Path("data/sherlock_clean.txt")
text = text_path.read_text(encoding="utf-8")

# 2. Split ratio
train_ratio = 0.9
split_idx = int(len(text) * train_ratio)

# 3. Split continuous text
train_text = text[:split_idx]
val_text = text[split_idx:]

# 4. Save datasets
train_path = Path("data/train_v4.txt")
val_path = Path("data/val_v4.txt")

train_path.write_text(train_text, encoding="utf-8")
val_path.write_text(val_text, encoding="utf-8")

# 5. Print statistics
print("--- MiniGPT V4 Dataset Split ---")
print("Total characters:", len(text))
print("Training characters:", len(train_text))
print("Validation characters:", len(val_text))

print(
    "Training ratio:",
    round(len(train_text) / len(text) * 100, 2), "%"
)

print(
    "Validation ratio:",
    round(len(val_text) / len(text) * 100, 2), "%"
)

print("\nSaved files:")
print(train_path)
print(val_path)

# 6. Verify no characters were lost
assert train_text + val_text == text

print("\nDataset split test passed!")
