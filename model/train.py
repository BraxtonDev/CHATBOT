
# LEM v3.0 - Training Engine

import os
import sys
import argparse
import torch

from tokenizer import encode
from transformer import LEMTransformer


# -----------------------------
# CONFIGURATION
# -----------------------------

CONFIG = {
    "vocab_size": 256,
    "block_size": 128,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1
}

BATCH_SIZE = 16
LEARNING_RATE = 0.0003
MAX_STEPS = 5000
EVAL_INTERVAL = 250
SAVE_INTERVAL = 500

DATA_PATH = os.path.join("data", "training.txt")
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "lem_latest.pt"
)


# -----------------------------
# DEVICE
# -----------------------------

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("LEM device:", device)


# -----------------------------
# LOAD DATA
# -----------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Training data not found: " + DATA_PATH
    )

with open(DATA_PATH, "r", encoding="utf-8") as file:
    text = file.read()

tokens = encode(text)

data = torch.tensor(tokens, dtype=torch.long)

if len(data) < CONFIG["block_size"] + 2:
    raise ValueError(
        "Training text is too short. Add more content to training.txt."
    )

split = int(len(data) * 0.9)

train_data = data[:split]
val_data = data[split:]

if len(val_data) < CONFIG["block_size"] + 1:
    val_data = train_data

print("Total tokens:", len(data))
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# -----------------------------
# BATCH GENERATOR
# -----------------------------

def get_batch(source):

    source = source.to(device)

    max_start = len(source) - CONFIG["block_size"] - 1

    if max_start < 1:
        raise ValueError("Not enough tokens for a training batch.")

    starts = torch.randint(
        0, max_start + 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[i:i + CONFIG["block_size"]]
        for i in starts.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + CONFIG["block_size"] + 1]
        for i in starts.tolist()
    ])

    return x.to(device), y.to(device)


# -----------------------------
# MODEL
# -----------------------------

model = LEMTransformer(CONFIG).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

step = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# RESUME TRAINING
# -----------------------------

if os.path.exists(CHECKPOINT_PATH):

    print("Found existing checkpoint.")

    try:
        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=device,
                weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=device
            )

        if checkpoint["config"] != CONFIG:
            raise ValueError(
                "Checkpoint architecture does not match CONFIG."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        step = checkpoint["step"]

        print("Resuming at step:", step)

    except Exception as error:
        raise RuntimeError(
            "Could not load checkpoint: " + str(error)
        )


# -----------------------------
# CHECKPOINT SAVING
# -----------------------------

def save_checkpoint():

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": CONFIG
    }

    torch.save(checkpoint, CHECKPOINT_PATH)

    print("Saved checkpoint:", CHECKPOINT_PATH)


# -----------------------------
# EVALUATION
# -----------------------------

@torch.no_grad()
def evaluate():

    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = []

        for _ in range(5):

            x, y = get_batch(source)
            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# -----------------------------
# TRAINING LOOP
# -----------------------------

print("\nStarting LEM training...")
print("Randomly initialized weights:", step == 0)

model.train()

while step < MAX_STEPS:

    x, y = get_batch(train_data)

    optimizer.zero_grad(set_to_none=True)

    _, loss = model(x, y)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )

    optimizer.step()

    step += 1

    if step % 25 == 0:
        print(
            f"Step {step}/{MAX_STEPS} | "
            f"Training loss: {loss.item():.4f}"
        )

    if step % EVAL_INTERVAL == 0:

        scores = evaluate()

        print(
            f"\nEvaluation at step {step}: "
            f"train={scores['train']:.4f}, "
            f"validation={scores['val']:.4f}\n"
        )

    if step % SAVE_INTERVAL == 0:
        save_checkpoint()


# -----------------------------
# FINISH
# -----------------------------

save_checkpoint()

print("\nLEM training session complete!")

