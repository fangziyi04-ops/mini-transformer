
from pathlib import Path
import re

# 1. Read original text
input_path = Path("data/sherlock.txt")
output_path = Path("data/sherlock_clean.txt")

text = input_path.read_text(encoding="utf-8-sig")

# 2. Remove Gutenberg header
start_pattern = r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"

start_match = re.search(
    start_pattern,
    text,
    flags=re.IGNORECASE
)

if not start_match:
    raise ValueError("Gutenberg START marker not found")

text = text[start_match.end():]

# 3. Remove Gutenberg footer
end_pattern = r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"

end_match = re.search(
    end_pattern,
    text,
    flags=re.IGNORECASE
)

if not end_match:
    raise ValueError("Gutenberg END marker not found")

text = text[:end_match.start()]

# 4. Normalize line endings and whitespace
text = text.replace("\r\n", "\n")
text = text.replace("\r", "\n")

# Remove excessive blank lines
text = re.sub(r"\n{3,}", "\n\n", text)

text = text.strip() + "\n"

# 5. Save cleaned text
output_path.write_text(text, encoding="utf-8")

# 6. Display results
print("--- Text Cleaning ---")
print("Original characters:",
      len(input_path.read_text(encoding="utf-8-sig")))
print("Cleaned characters:", len(text))
print("Removed characters:",
      len(input_path.read_text(encoding="utf-8-sig")) - len(text))

print("\n--- First 500 Characters ---")
print(text[:500])

print("\nSaved to:", output_path)
