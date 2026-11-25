
# LEM v3.0 - Text Generation

import os
import argparse
import torch

from tokenizer import encode, decode
from transformer import LEMTransformer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    default="Hello, I am LEM."
)

parser.add_argument(
    "--tokens",
    type=int,
    default=200
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.8
)

parser.add_argument(
    "--top_k",
    type=int,
    default=40
)

args = parser.parse_args()


# -----------------------------
# DEVICE
# -----------------------------

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# -----------------------------
# LOAD CHECKPOINT
# -----------------------------

checkpoint_path = os.path.join(
    "checkpoints", "lem_latest.pt"
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        "No trained model found. Run model/train.py first."
    )

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
except TypeError:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


config = checkpoint["config"]

model = LEMTransformer(config).to(device)

model.load_state_dict(checkpoint["model"])

model.eval()

print("Loaded LEM checkpoint.")
print("Training step:", checkpoint["step"])
print("Device:", device)


# -----------------------------
# GENERATION
# -----------------------------

@torch.no_grad()
def generate(prompt, max_tokens=200):

    tokens = encode(prompt)

    if not tokens:
        tokens = [32]

    context = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_tokens):

        context = context[:, -config["block_size"]:]

        logits, _ = model(context)

        logits = logits[:, -1, :]

        temperature = max(args.temperature, 0.01)
        logits = logits / temperature

        if args.top_k > 0:

            k = min(args.top_k, logits.size(-1))

            values, _ = torch.topk(logits, k)

            threshold = values[:, [-1]]

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf")
            )

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        context = torch.cat(
            (context, next_token),
            dim=1
        )

    return decode(context[0].tolist())


# -----------------------------
# RUN
# -----------------------------

print("\nPrompt:", args.prompt)
print("\nLEM is generating...\n")

result = generate(
    args.prompt,
    args.tokens
)

print("========== LEM OUTPUT ==========\n")
print(result)
print("\n================================")

