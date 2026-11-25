"""
LEM - From-scratch Transformer training script

Features:
- Randomly initialized Transformer weights
- UTF-8 byte tokenizer
- Automatic CPU / CUDA / Apple MPS selection
- Training and validation loss tracking
- Saves training_stats.json for stats.html
- Saves checkpoints/lem_latest.pt
- Resumes training from the latest checkpoint

Run from the project root:
    python model/train.py
"""

import json
import os
import random
from datetime import datetime, timezone

import torch

from tokenizer import encode
from transformer import LEMTransformer


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

CONFIG = {
    "vocab_size": 256,
    "block_size": 128,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.1,
}


# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------

BATCH_SIZE = 16
LEARNING_RATE = 0.0003

# Increase this number to train for more steps.
MAX_STEPS = 5000

EVAL_INTERVAL = 250
SAVE_INTERVAL = 500
EVAL_BATCHES = 5


# --------------------------------------------------
# FILE LOCATIONS
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT, "data", "training.txt")

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "lem_latest.pt")

STATS_PATH = os.path.join(ROOT, "training_stats.json")


# --------------------------------------------------
# DEVICE SELECTION
# --------------------------------------------------

def choose_device():

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

def load_stats():

    if os.path.exists(STATS_PATH):

        try:

            with open(STATS_PATH, "r", encoding="utf-8") as file:
                old_stats = json.load(file)

            return {
                "step": int(old_stats.get("step", 0)),
                "max_steps": MAX_STEPS,
                "device": old_stats.get("device", "unknown"),
                "train_loss": old_stats.get("train_loss", []),
                "val_loss": old_stats.get("val_loss", []),
                "updated_at": old_stats.get("updated_at"),
            }

        except (
            json.JSONDecodeError,
            OSError,
            TypeError,
            ValueError,
        ):
            print("Could not read previous statistics. Starting new stats.")

    return {
        "step": 0,
        "max_steps": MAX_STEPS,
        "device": "unknown",
        "train_loss": [],
        "val_loss": [],
        "updated_at": None,
    }


def save_stats(stats, step, device):

    stats["step"] = step
    stats["max_steps"] = MAX_STEPS
    stats["device"] = str(device)

    stats["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    temporary_path = STATS_PATH + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as file:

        json.dump(
            stats,
            file,
            indent=2,
        )

    os.replace(temporary_path, STATS_PATH)


# --------------------------------------------------
# BATCH GENERATION
# --------------------------------------------------

def get_batch(
    split,
    train_data,
    val_data,
    device,
):

    if split == "train":
        source = train_data
    else:
        source = val_data

    block_size = CONFIG["block_size"]

    if len(source) <= block_size:

        raise ValueError(
            f"The {split} dataset is too small. "
            f"It contains {len(source)} tokens, but needs "
            f"more than {block_size}. Add more text to "
            f"data/training.txt."
        )

    starts = torch.randint(
        0,
        len(source) - block_size,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in starts.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in starts.tolist()
    ])

    return x.to(device), y.to(device)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    train_data,
    val_data,
    device,
):

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = []

        for _ in range(EVAL_BATCHES):

            x, y = get_batch(
                split,
                train_data,
                val_data,
                device,
            )

            _, loss = model(x, y)

            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()

    return results


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    random.seed(42)
    torch.manual_seed(42)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # Load training text.

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Training file not found: {DATA_PATH}"
        )

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        raw_text = file.read()

    if not raw_text.strip():

        raise ValueError(
            "training.txt is empty."
        )

    # Convert text into byte tokens.

    token_ids = encode(raw_text)

    if len(token_ids) < CONFIG["block_size"] * 2:

        raise ValueError(
            "The training dataset is too small. "
            "Add more text to data/training.txt."
        )

    all_data = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    # Split into training and validation data.

    split_index = int(0.9 * len(all_data))

    train_data = all_data[:split_index]
    val_data = all_data[split_index:]

    # Select device.

    device = choose_device()

    print()
    print("================================")
    print("          LEM TRAINING")
    print("================================")
    print(f"Device: {device}")
    print(f"Total tokens: {len(all_data):,}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Maximum steps: {MAX_STEPS}")
    print("================================")
    print()

    # Create model.

    model = LEMTransformer(CONFIG).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    stats = load_stats()

    start_step = 0

    # --------------------------------------------------
    # RESUME FROM CHECKPOINT
    # --------------------------------------------------

    if os.path.exists(CHECKPOINT_PATH):

        print("Loading previous checkpoint...")

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

        if checkpoint.get("config") != CONFIG:

            raise ValueError(
                "The checkpoint configuration does not match "
                "the current CONFIG. Use the original configuration "
                "or move the old checkpoint out of checkpoints/."
            )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        start_step = int(
            checkpoint.get("step", 0)
        )

        print(f"Resuming from step {start_step}")

    # --------------------------------------------------
    # CHECKPOINT SAVING
    # --------------------------------------------------

    def save_checkpoint(step):

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": CONFIG,
            },
            CHECKPOINT_PATH,
        )

        print(
            f"Checkpoint saved at step {step}"
        )

    # Check whether training is already complete.

    if start_step >= MAX_STEPS:

        print(
            "The checkpoint has already reached MAX_STEPS."
        )

        print(
            "Increase MAX_STEPS in train.py to continue training."
        )

        save_stats(
            stats,
            start_step,
            device,
        )

        return

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------

    model.train()

    for step in range(
        start_step + 1,
        MAX_STEPS + 1,
    ):

        # Get a batch.

        x, y = get_batch(
            "train",
            train_data,
            val_data,
            device,
        )

        # Forward pass.

        _, loss = model(x, y)

        # Backpropagation.

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # Prevent excessively large gradients.

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        # --------------------------------------------------
        # PRINT TRAINING PROGRESS
        # --------------------------------------------------

        if step % 25 == 0:

            print(
                f"Step {step}/{MAX_STEPS} "
                f"| Batch loss: {loss.item():.4f}"
            )

            save_stats(
                stats,
                step,
                device,
            )

        # --------------------------------------------------
        # EVALUATE MODEL
        # --------------------------------------------------

        if (
            step % EVAL_INTERVAL == 0
            or step == 1
        ):

            scores = evaluate(
                model,
                train_data,
                val_data,
                device,
            )

            train_loss = scores["train"]
            val_loss = scores["val"]

            print()
            print(f"Evaluation at step {step}")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {val_loss:.4f}")
            print()

            # Add data for stats.html.

            stats["train_loss"].append({
                "step": step,
                "loss": train_loss,
            })

            stats["val_loss"].append({
                "step": step,
                "loss": val_loss,
            })

            save_stats(
                stats,
                step,
                device,
            )

        # --------------------------------------------------
        # SAVE CHECKPOINT
        # --------------------------------------------------

        if step % SAVE_INTERVAL == 0:

            save_checkpoint(step)

    # --------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------

    save_checkpoint(MAX_STEPS)

    save_stats(
        stats,
        MAX_STEPS,
        device,
    )

    print()
    print("================================")
    print("       TRAINING FINISHED")
    print("================================")
    print(f"Final step: {MAX_STEPS}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Statistics: {STATS_PATH}")
    print()


if __name__ == "__main__":
    main()
