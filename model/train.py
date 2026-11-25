
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import torch

from tokenizer import VOCAB_SIZE
from transformer import LEMTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "vocab_size": VOCAB_SIZE,
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
EVAL_BATCHES = 5

TRAIN_SPLIT = 0.90

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

CACHE_PATH = DATA_DIR / "training.bin"
CACHE_INFO_PATH = DATA_DIR / "training_cache_info.json"

CHECKPOINT_PATH = CHECKPOINT_DIR / "lem_latest.pt"
STATS_PATH = ROOT / "training_stats.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 60)
print("LEM TRAINING SYSTEM")
print("=" * 60)
print("Device:", DEVICE)
print("PyTorch:", torch.__version__)


# ============================================================
# FIND DATASET FILES
# ============================================================

def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_dataset_files():
    part_files = list(DATA_DIR.glob("training_part*.txt"))
    part_files = sorted(part_files, key=natural_sort_key)

    if part_files:
        return part_files

    single_file = DATA_DIR / "training.txt"

    if single_file.exists():
        return [single_file]

    raise FileNotFoundError(
        "No training data found.\n"
        "Expected files such as:\n"
        "data/training_part1.txt\n"
        "data/training_part2.txt\n"
        "data/training_part3.txt\n"
        "\nOr a single data/training.txt file."
    )


# ============================================================
# BUILD BYTE TOKEN CACHE
# ============================================================

BOS = b"<|bos|>"
EOS = b"<|eos|>"

def cache_is_current(files):
    if not CACHE_PATH.exists() or not CACHE_INFO_PATH.exists():
        return False

    try:
        with open(CACHE_INFO_PATH, "r", encoding="utf-8") as f:
            old_info = json.load(f)

        current_info = [
            {
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime_ns
            }
            for file in files
        ]

        return old_info.get("sources") == current_info

    except Exception:
        return False


def write_cleaned_file(source, output):
    """
    Copy a text file into the binary cache while removing
    BOS and EOS markers.

    Uses chunks so the whole file is never loaded into RAM.
    """

    carry = b""
    chunk_size = 1024 * 1024
    keep_length = max(len(BOS), len(EOS)) - 1

    with open(source, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            data = carry + chunk

            if len(data) <= keep_length:
                carry = data
                continue

            safe_length = len(data) - keep_length

            safe_data = data[:safe_length]
            carry = data[safe_length:]

            safe_data = safe_data.replace(BOS, b"")
            safe_data = safe_data.replace(EOS, b"")

            output.write(safe_data)

    if carry:
        carry = carry.replace(BOS, b"")
        carry = carry.replace(EOS, b"")
        output.write(carry)


def build_dataset_cache(files):
    print("\nBuilding training cache...")
    print("Source files:", len(files))

    temporary_path = CACHE_PATH.with_suffix(".tmp")

    try:
        with open(temporary_path, "wb") as output:
            for index, source in enumerate(files):
                print(
                    f"Processing {index + 1}/{len(files)}: "
                    f"{source.name}"
                )

                if index > 0:
                    output.write(b"\n\n")

                write_cleaned_file(source, output)

        os.replace(temporary_path, CACHE_PATH)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    source_info = [
        {
            "name": file.name,
            "size": file.stat().st_size,
            "modified": file.stat().st_mtime_ns
        }
        for file in files
    ]

    info = {
        "sources": source_info,
        "cache_size": CACHE_PATH.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    temporary_info = CACHE_INFO_PATH.with_suffix(".tmp")

    with open(temporary_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    os.replace(temporary_info, CACHE_INFO_PATH)

    print("Cache created:", CACHE_PATH)
    print("Cache size:", CACHE_PATH.stat().st_size, "bytes")


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    files = find_dataset_files()

    print("\nDataset files:")

    for file in files:
        print(
            " -",
            file.name,
            "|",
            round(file.stat().st_size / 1024 / 1024, 2),
            "MB"
        )

    if cache_is_current(files):
        print("\nExisting training cache is current.")
    else:
        build_dataset_cache(files)

    data_size = CACHE_PATH.stat().st_size

    if data_size <= CONFIG["block_size"] + 10:
        raise ValueError(
            "The training dataset is too small. "
            "Add more text before training."
        )

    # Memory-mapped byte tensor.
    # This avoids loading the complete dataset into a Python list.
    data = torch.from_file(
        str(CACHE_PATH),
        shared=False,
        size=data_size,
        dtype=torch.uint8
    )

    print("\nTotal training bytes:", f"{data_size:,}")
    print("Dataset loaded using memory mapping.")

    return data


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

data = load_dataset()

split_index = int(len(data) * TRAIN_SPLIT)

train_data = data[:split_index]
val_data = data[split_index:]

if len(val_data) <= CONFIG["block_size"]:
    raise ValueError("Not enough validation data.")

if len(train_data) <= CONFIG["block_size"]:
    raise ValueError("Not enough training data.")

print("\nTraining bytes:", f"{len(train_data):,}")
print("Validation bytes:", f"{len(val_data):,}")


# ============================================================
# MODEL
# ============================================================

model = LEMTransformer(CONFIG).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:", f"{parameter_count:,}")


# ============================================================
# BATCH GENERATOR
# ============================================================

def get_batch(source):
    max_start = len(source) - CONFIG["block_size"] - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (BATCH_SIZE,)
    )

    offsets = torch.arange(CONFIG["block_size"])

    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

    x = source[indices].to(
        DEVICE,
        dtype=torch.long
    )

    y = source[indices + 1].to(
        DEVICE,
        dtype=torch.long
    )

    return x, y


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("val", val_data)
    ]:
        losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(source)

            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# ============================================================
# TRAINING STATISTICS
# ============================================================

stats = {
    "project": "LEM",
    "version": "2.5",
    "step": 0,
    "max_steps": MAX_STEPS,
    "device": DEVICE,
    "dataset_files": [],
    "dataset_bytes": len(data),
    "parameters": parameter_count,
    "train_loss": None,
    "val_loss": None,
    "loss_history": [],
    "started_at": datetime.now(timezone.utc).isoformat(),
    "updated_at": None,
    "status": "starting"
}

for file in find_dataset_files():
    stats["dataset_files"].append(file.name)


def save_stats():
    stats["updated_at"] = datetime.now(timezone.utc).isoformat()

    temporary_path = STATS_PATH.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    os.replace(temporary_path, STATS_PATH)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

start_step = 0

if CHECKPOINT_PATH.exists():
    print("\nFound existing checkpoint.")

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        if checkpoint.get("config") != CONFIG:
            print("Checkpoint configuration differs.")
            print("Starting a new training run.")
        else:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            start_step = checkpoint.get("step", 0)

            print("Checkpoint loaded.")
            print("Resuming from step:", start_step)

    except Exception as error:
        print("Could not load checkpoint:", error)
        print("Starting from the beginning.")


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(step):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": CONFIG
    }

    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, CHECKPOINT_PATH)

    print("Checkpoint saved:", CHECKPOINT_PATH)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

model.train()

stats["status"] = "training"
save_stats()

start_time = time.time()

try:
    for step in range(start_step + 1, MAX_STEPS + 1):

        x, y = get_batch(train_data)

        optimizer.zero_grad(set_to_none=True)

        logits, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        stats["step"] = step
        stats["train_loss"] = float(loss.item())

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        if step % EVAL_INTERVAL == 0 or step == 1:
            losses = estimate_loss()

            stats["train_loss"] = losses["train"]
            stats["val_loss"] = losses["val"]

            stats["loss_history"].append({
                "step": step,
                "train": losses["train"],
                "validation": losses["val"]
            })

            elapsed = time.time() - start_time

            print(
                f"Step {step}/{MAX_STEPS} | "
                f"Train: {losses['train']:.4f} | "
                f"Validation: {losses['val']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            save_stats()

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if step % SAVE_INTERVAL == 0:
            save_checkpoint(step)
            save_stats()

    # Final checkpoint
    save_checkpoint(MAX_STEPS)

    stats["status"] = "completed"
    stats["step"] = MAX_STEPS

    save_stats()

    print("\nTraining completed!")

except KeyboardInterrupt:
    print("\nTraining interrupted.")

    save_checkpoint(stats["step"])

    stats["status"] = "interrupted"
    save_stats()

except Exception as error:
    print("\nTraining error:", error)

    stats["status"] = "error"
    stats["error"] = str(error)

    save_stats()

    raise

finally:
    elapsed = time.time() - start_time

    print("\nTraining time:", round(elapsed, 2), "seconds")
    print("Current step:", stats["step"])
    print("Status:", stats["status"])
    print("Stats:", STATS_PATH)
