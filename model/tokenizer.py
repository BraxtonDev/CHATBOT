
# LEM v3.0 - Byte Tokenizer

VOCAB_SIZE = 256

def encode(text):
    """Convert text into UTF-8 byte token IDs."""
    return list(text.encode("utf-8"))

def decode(tokens):
    """Convert byte token IDs back into text."""
    return bytes(
        max(0, min(255, int(token))) for token in tokens
    ).decode("utf-8", errors="replace")

