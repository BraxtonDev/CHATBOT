
# model/finetune.py
# LEM v3.0 - Conversation Fine-Tuning

import json
import random
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import torch

from tokenizer import encode
from transformer import LEMTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
MAX_STEPS = 2000

EVAL_INTERVAL = 100
SAVE_INTERVAL = 250
EVAL_BATCHES = 10

# The pretrained checkpoint supplies the model configuration.
# Do not change its vocabulary or architecture here.

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

TRAIN_FILE = DATA_DIR / "conversations_train.txt"
VALIDATION_FILE = DATA_DIR / "conversations_validation.txt"

PRETRAINED_CHECKPOINT = CHECKPOINT_DIR / "lem_latest.pt"
CHAT_CHECKPOINT = CHECKPOINT_DIR / "lem_chat_latest.pt"

STATS_PATH = ROOT / "finetuning_stats.json"

BOS = "<|bos|>"
EOS = "<|eos|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("=" * 60)
print("LEM v3.0 CONVERSATION FINE-TUNING")
print("=" * 60)
print("Device:", DEVICE)
print("PyTorch:", torch.__version__)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

if not PRETRAINED_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Pretrained checkpoint not found:\n{PRETRAINED_CHECKPOINT}\n\n"
        "Run your original training script first."
    )

print("\nLoading pretrained checkpoint...")

checkpoint = torch.load(
    PRETRAINED_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if "model" not in checkpoint or "config" not in checkpoint:
    raise ValueError(
        "Checkpoint must contain 'model' and 'config'."
    )

CONFIG = checkpoint["config"]

if CONFIG.get("vocab_size") != 256:
    raise ValueError(
        "This fine-tuning script expects the existing 256-byte tokenizer."
    )

BLOCK_SIZE = CONFIG["block_size"]

model = LEMTransformer(CONFIG).to(DEVICE)
model.load_state_dict(checkpoint["model"])

print("Pretrained model loaded.")
print("Starting from checkpoint step:", checkpoint.get("step", 0))
print("Context length:", BLOCK_SIZE)
print("Vocabulary size:", CONFIG["vocab_size"])


# ============================================================
# LOAD AND VALIDATE CONVERSATIONS
# ============================================================

def load_conversations(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Conversation file not found: {path}\n"
            "Run: python model/prepare_data.py"
        )

    text = path.read_text(encoding="utf-8-sig")

    pattern = re.compile(
        r"<\|bos\|>.*?<\|eos\|>",
        re.DOTALL
    )

    conversations = [
        match.group(0).strip()
        for match in pattern.finditer(text)
    ]

    if not conversations:
        raise ValueError(f"No conversations found in {path}")

    return conversations


# ============================================================
# CREATE ASSISTANT-ONLY TARGET MASKS
# ============================================================

def make_example(conversation):
    """
    Convert one conversation into UTF-8 bytes.

    The input contains the entire conversation context.
    Loss is calculated only for assistant markers, assistant
    responses, and the final EOS marker.
    """

    raw = conversation.encode("utf-8")
    target_mask = bytearray(len(raw))

    assistant_marker = ASSISTANT.encode("utf-8")
    user_marker = USER.encode("utf-8")
    eos_marker = EOS.encode("utf-8")

    position = 0

    while True:
        assistant_start = raw.find(assistant_marker, position)

        if assistant_start == -1:
            break

        response_start = assistant_start
        response_end = len(raw)

        next_user = raw.find(user_marker, response_start)
        next_eos = raw.find(eos_marker, response_start)

        possible_ends = [
            index for index in (next_user, next_eos)
            if index != -1
        ]

        if possible_ends:
            response_end = min(possible_ends)

        # Include assistant marker and response bytes.
        for index in range(response_start, response_end):
            target_mask[index] = 1

        # Also train the model to predict EOS.
        if next_eos != -1 and next_eos == response_end:
            for index in range(next_eos, next_eos + len(eos_marker)):
                target_mask[index] = 1

        position = max(response_end, assistant_start + len(assistant_marker))

    if len(raw) < 2:
        return None

    if not any(target_mask):
        return None

    return raw, target_mask


def prepare_dataset(conversations, name):
    examples = []

    for conversation in conversations:
        example = make_example(conversation)

        if example is not None:
            examples.append(example)

    if not examples:
        raise ValueError(
            f"No usable assistant responses found in {name}."
        )

    print(f"{name}: {len(examples)} usable conversations")

    return examples


train_conversations = load_conversations(TRAIN_FILE)
validation_conversations = load_conversations(VALIDATION_FILE)

train_examples = prepare_dataset(
    train_conversations,
    "Training"
)

validation_examples = prepare_dataset(
    validation_conversations,
    "Validation"
)


# ============================================================
# RANDOM TRAINING BATCHES
# ============================================================

def get_batch(examples):
    """
    Randomly sample context windows.

    Input bytes are used as context.
    Only assistant response target bytes contribute to loss.
    """

    xs = []
    ys = []

    for _ in range(BATCH_SIZE):
        selected = None

        # Retry until the sampled window includes assistant text.
        for attempt in range(100):
            raw, mask = random.choice(examples)

            max_start = max(0, len(raw) - BLOCK_SIZE - 1)

            start = random.randint(0, max_start)

            end = min(start + BLOCK_SIZE + 1, len(raw))

            window_mask = mask[start + 1:end]

            if any(window_mask):
                selected = (raw, mask, start, end)
                break

        if selected is None:
            raise RuntimeError(
                "Could not find a window containing assistant targets. "
                "Check the conversation dataset."
            )

        raw, mask, start, end = selected

        x_bytes = raw[start:end - 1]
        y_bytes = raw[start + 1:end]

        target_mask = mask[start + 1:end]

        x = list(x_bytes)
        y = list(y_bytes)

        # Pad shorter conversations to the model context length.
        padding = BLOCK_SIZE - len(x)

        if padding > 0:
            x.extend([0] * padding)
            y.extend([-100] * padding)
            target_mask.extend([0] * padding)

        # Ignore user-message bytes when calculating loss.
        for index, should_train in enumerate(target_mask):
            if not should_train:
                y[index] = -100

        xs.append(x)
        ys.append(y)

    x_tensor = torch.tensor(
        xs,
        dtype=torch.long,
        device=DEVICE
    )

    y_tensor = torch.tensor(
        ys,
        dtype=torch.long,
        device=DEVICE
    )

    return x_tensor, y_tensor


# ============================================================
# LOSS EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for name, examples in [
        ("train", train_examples),
        ("validation", validation_examples)
    ]:
        losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(examples)

            _, loss = model(x, y)

            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())

        results[name] = (
            sum(losses) / len(losses)
            if losses else None
        )

    model.train()

    return results


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nTrainable parameters:", f"{parameter_count:,}")
print("Training conversations:", len(train_examples))
print("Validation conversations:", len(validation_examples))


# ============================================================
# STATISTICS
# ============================================================

stats = {
    "project": "LEM",
    "version": "3.0",
    "stage": "conversation_finetuning",
    "step": 0,
    "max_steps": MAX_STEPS,
    "device": DEVICE,
    "parameters": parameter_count,
    "training_conversations": len(train_examples),
    "validation_conversations": len(validation_examples),
    "train_loss": None,
    "validation_loss": None,
    "loss_history": [],
    "started_at": datetime.now(timezone.utc).isoformat(),
    "updated_at": None,
    "status": "starting"
}


def save_stats():
    stats["updated_at"] = datetime.now(timezone.utc).isoformat()

    temporary_path = STATS_PATH.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    temporary_path.replace(STATS_PATH)


# ============================================================
# SAVE CHAT CHECKPOINT
# ============================================================

def save_checkpoint(step):
    checkpoint_data = {
        "model": model.state_dict(),
        "step": step,
        "config": CONFIG,
        "stage": "conversation_finetuning",
        "source_checkpoint": PRETRAINED_CHECKPOINT.name
    }

    temporary_path = CHAT_CHECKPOINT.with_suffix(".tmp")

    torch.save(checkpoint_data, temporary_path)
    temporary_path.replace(CHAT_CHECKPOINT)

    print("Chat checkpoint saved:", CHAT_CHECKPOINT)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING CONVERSATION FINE-TUNING")
print("=" * 60)

model.train()

stats["status"] = "training"
save_stats()

start_time = time.time()

try:
    for step in range(1, MAX_STEPS + 1):

        x, y = get_batch(train_examples)

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Invalid training loss.")

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
            stats["validation_loss"] = losses["validation"]

            stats["loss_history"].append({
                "step": step,
                "train": losses["train"],
                "validation": losses["validation"]
            })

            elapsed = time.time() - start_time

            print(
                f"Step {step}/{MAX_STEPS} | "
                f"Train: {losses['train']:.4f} | "
                f"Validation: {losses['validation']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            save_stats()

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if step % SAVE_INTERVAL == 0:
            save_checkpoint(step)
            save_stats()

    save_checkpoint(MAX_STEPS)

    stats["status"] = "completed"
    stats["step"] = MAX_STEPS
    save_stats()

    print("\nFine-tuning completed!")

except KeyboardInterrupt:
    print("\nFine-tuning interrupted.")

    save_checkpoint(stats["step"])

    stats["status"] = "interrupted"
    save_stats()

except Exception as error:
    print("\nFine-tuning error:", error)

    stats["status"] = "error"
    stats["error"] = str(error)

    save_stats()

    raise

finally:
    elapsed = time.time() - start_time

    print("\nFine-tuning time:", round(elapsed, 2), "seconds")
    print("Current step:", stats["step"])
    print("Status:", stats["status"])
    print("Chat checkpoint:", CHAT_CHECKPOINT)
    print("Statistics:", STATS_PATH)
