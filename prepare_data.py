
import random

random.seed(42)

# Read original text
with open("input.txt", "r", encoding="utf-8") as f:
    original_text = f.read()

sentences = [
    line.strip()
    for line in original_text.splitlines()
    if line.strip()
]

# Create a larger synthetic dataset
expanded_sentences = []

for _ in range(200):
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    expanded_sentences.extend(shuffled)

expanded_text = "\n".join(expanded_sentences) + "\n"

# Save separately to preserve original data
with open("input_v3.txt", "w", encoding="utf-8") as f:
    f.write(expanded_text)

print("--- Dataset Preparation ---")
print("Original characters:", len(original_text))
print("Expanded characters:", len(expanded_text))
print("Number of sentences:", len(expanded_sentences))
print("Saved to input_v3.txt")
