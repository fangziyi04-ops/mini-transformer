
import json
from pathlib import Path

# 1. Load cleaned text
text_path = Path("data/sherlock_clean.txt")
text = text_path.read_text(encoding="utf-8")

# 2. Build character vocabulary
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 3. Encoding and decoding
def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# 4. Save V4 vocabulary
vocab_path = Path("data/vocab_v4.json")

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(
        {"chars": chars},
        f,
        ensure_ascii=False,
        indent=2
    )

# 5. Test tokenizer
test_text = "Sherlock Holmes"

encoded = encode(test_text)
decoded = decode(encoded)

print("--- MiniGPT V4 Tokenizer ---")
print("Total characters:", len(text))
print("Vocabulary size:", vocab_size)
print("Test text:", test_text)
print("Encoded:", encoded)
print("Decoded:", decoded)
print("Vocabulary saved to:", vocab_path)

assert decoded == test_text
print("Tokenizer test passed!")
