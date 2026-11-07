
import json

# 1. Read training text
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Build character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 3. Create character-to-ID mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 4. Encode and decode
def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# 5. Save vocabulary
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        {"chars": chars},
        f,
        ensure_ascii=False,
        indent=2
    )

# 6. Test tokenizer
print("\n--- Tokenizer Test ---")
print("Total characters:", len(text))
print("Vocabulary size:", vocab_size)
print("Vocabulary:", chars)

test_text = "Transformers"

encoded = encode(test_text)
decoded = decode(encoded)

print("\nOriginal:", test_text)
print("Encoded:", encoded)
print("Decoded:", decoded)
print("\nVocabulary saved to vocab.json")
