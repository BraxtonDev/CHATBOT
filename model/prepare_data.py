# model/prepare_data.py

from pathlib import Path
import argparse
import random
import re
import sys


# ---------------------------------------------------------
# LEM CONVERSATION DATA PREPARATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

TRAIN_OUTPUT = DATA_DIR / "conversations_train.txt"
VALIDATION_OUTPUT = DATA_DIR / "conversations_validation.txt"

BOS = "<|bos|>"
EOS = "<|eos|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

CONVERSATION_PATTERN = re.compile(
    r"<\|bos\|>.*?<\|eos\|>",
    re.DOTALL
)


def normalize_for_comparison(text):
    """Normalize whitespace for duplicate detection."""
    return " ".join(text.split())


def find_conversations(file_path):
    """Read and extract complete conversations from one text file."""

    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        print(f"[ERROR] Could not decode: {file_path}")
        return [], 1
    except OSError as error:
        print(f"[ERROR] Could not read {file_path}: {error}")
        return [], 1

    matches = list(CONVERSATION_PATTERN.finditer(text))
    conversations = []

    # Check that there is no non-whitespace text outside
    # the complete BOS/EOS conversation blocks.
    remaining = CONVERSATION_PATTERN.sub("", text)

    if remaining.strip():
        print(
            f"[WARNING] Text outside conversation markers "
            f"in {file_path.name}"
        )

    errors = 0

    for number, match in enumerate(matches, start=1):
        conversation = match.group(0).strip()

        if USER not in conversation:
            print(
                f"[WARNING] Conversation {number} in "
                f"{file_path.name} has no user message."
            )
            errors += 1
            continue

        if ASSISTANT not in conversation:
            print(
                f"[WARNING] Conversation {number} in "
                f"{file_path.name} has no assistant response."
            )
            errors += 1
            continue

        # Check that user and assistant messages alternate.
        message_markers = re.findall(
            r"<\|user\|>|<\|assistant\|>",
            conversation
        )

        if not message_markers or message_markers[0] != USER:
            print(
                f"[WARNING] Conversation {number} in "
                f"{file_path.name} does not start with a user message."
            )
            errors += 1
            continue

        valid_order = True

        for index, marker in enumerate(message_markers):
            expected = USER if index % 2 == 0 else ASSISTANT

            if marker != expected:
                valid_order = False
                break

        if not valid_order:
            print(
                f"[WARNING] Conversation {number} in "
                f"{file_path.name} has incorrectly ordered messages."
            )
            errors += 1
            continue

        conversations.append(conversation)

    if not matches:
        print(f"[WARNING] No conversations found in {file_path.name}")

    return conversations, errors


def load_all_conversations():
    """Load conversations from every TXT file in the input folder."""

    if not CONVERSATIONS_DIR.exists():
        print(f"[ERROR] Folder not found: {CONVERSATIONS_DIR}")
        print("Create data/conversations/ and put your TXT files inside.")
        sys.exit(1)

    files = sorted(
        CONVERSATIONS_DIR.rglob("*.txt"),
        key=lambda path: str(path).lower()
    )

    if not files:
        print("[ERROR] No TXT files found in data/conversations/")
        sys.exit(1)

    print()
    print("LEM CONVERSATION DATA PREPARATION")
    print("=" * 45)
    print(f"Input folder: {CONVERSATIONS_DIR}")
    print(f"TXT files found: {len(files)}")
    print()

    all_conversations = []
    total_errors = 0

    for file_path in files:
        conversations, errors = find_conversations(file_path)

        print(
            f"{file_path.name}: "
            f"{len(conversations)} valid conversations"
        )

        all_conversations.extend(conversations)
        total_errors += errors

    return all_conversations, total_errors


def remove_duplicates(conversations):
    """Remove exact duplicate conversations."""

    unique = []
    seen = set()

    for conversation in conversations:
        comparison = normalize_for_comparison(conversation)

        if comparison not in seen:
            seen.add(comparison)
            unique.append(conversation)

    return unique, len(conversations) - len(unique)


def save_conversations(file_path, conversations):
    """Save conversations in the LEM text format."""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    content = "\n\n".join(conversations)

    if content:
        content += "\n"

    file_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LEM conversation training data."
    )

    parser.add_argument(
        "--validation",
        type=float,
        default=0.10,
        help="Fraction used for validation. Default: 0.10"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for a repeatable split. Default: 42"
    )

    args = parser.parse_args()

    if not 0 <= args.validation < 1:
        print("[ERROR] Validation fraction must be at least 0 and less than 1.")
        sys.exit(1)

    conversations, errors = load_all_conversations()

    if not conversations:
        print()
        print("[ERROR] No valid conversations were found.")
        sys.exit(1)

    print()
    print("Removing duplicates...")

    conversations, duplicates = remove_duplicates(conversations)

    if not conversations:
        print("[ERROR] No unique conversations remain.")
        sys.exit(1)

    # Shuffle deterministically so the same data and seed
    # produce the same training/validation split.
    random_generator = random.Random(args.seed)
    random_generator.shuffle(conversations)

    total = len(conversations)

    if total >= 2 and args.validation > 0:
        validation_count = max(
            1,
            round(total * args.validation)
        )
        validation_count = min(validation_count, total - 1)
    else:
        validation_count = 0

    training_count = total - validation_count

    training_data = conversations[:training_count]
    validation_data = conversations[training_count:]

    save_conversations(TRAIN_OUTPUT, training_data)
    save_conversations(VALIDATION_OUTPUT, validation_data)

    print()
    print("=" * 45)
    print("PREPARATION COMPLETE")
    print("=" * 45)
    print(f"Valid conversations: {total + duplicates}")
    print(f"Duplicates removed:  {duplicates}")
    print(f"Invalid conversations: {errors}")
    print(f"Training conversations: {len(training_data)}")
    print(f"Validation conversations: {len(validation_data)}")
    print()
    print(f"Training file:   {TRAIN_OUTPUT}")
    print(f"Validation file: {VALIDATION_OUTPUT}")
    print()
    print("Next step: use these files with LEM's fine-tuning script.")


if __name__ == "__main__":
    main()
