
# model/evaluate.py
# LEM v3.0 - Interactive Chatbot Testing

import torch
from pathlib import Path

from tokenizer import encode, decode
from transformer import LEMTransformer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "lem_chat_latest.pt"

MAX_NEW_TOKENS = 400
TEMPERATURE = 0.8
TOP_K = 40

BOS = "<|bos|>"
EOS = "<|eos|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"


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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Chat checkpoint not found:\n{CHECKPOINT_PATH}\n\n"
            "Run model/finetune.py first."
        )

    print("Loading LEM chatbot...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    config = checkpoint["config"]

    model = LEMTransformer(config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("LEM loaded successfully.")
    print("Device:", DEVICE)
    print("Context length:", config["block_size"])
    print("Training step:", checkpoint.get("step", "Unknown"))

    return model, config


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_response(model, config, prompt):
    conversation = (
        BOS
        + USER
        + prompt
        + "\n"
        + ASSISTANT
    )

    tokens = encode(conversation)

    generated = []

    eos_bytes = EOS.encode("utf-8")
    user_bytes = USER.encode("utf-8")

    for _ in range(MAX_NEW_TOKENS):
        context = tokens[-config["block_size"]:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )

        logits, _ = model(x)

        next_logits = logits[0, -1, :] / TEMPERATURE

        # Top-k sampling.
        if TOP_K > 0:
            values, _ = torch.topk(
                next_logits,
                min(TOP_K, next_logits.numel())
            )

            cutoff = values[-1]

            next_logits[next_logits < cutoff] = float("-inf")

        probabilities = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        tokens.append(next_token)
        generated.append(next_token)

        generated_bytes = bytes(generated)

        # Stop at the end of the assistant response.
        if generated_bytes.endswith(eos_bytes):
            generated = generated[:-len(eos_bytes)]
            break

        # Avoid continuing into another user message.
        if generated_bytes.endswith(user_bytes):
            generated = generated[:-len(user_bytes)]
            break

    response = decode(generated)

    # Remove any accidental formatting markers.
    response = response.replace(EOS, "")
    response = response.replace(USER, "")
    response = response.strip()

    return response


# ============================================================
# CHAT LOOP
# ============================================================

def main():
    print("=" * 55)
    print("LEM v3.0 CHATBOT")
    print("=" * 55)

    model, config = load_model()

    print("\nType a message to chat with LEM.")
    print("Commands: /exit, /quit")
    print("-" * 55)

    while True:
        try:
            prompt = input("\nYou: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ("/exit", "/quit"):
                print("\nClosing LEM.")
                break

            print("\nLEM: ", end="", flush=True)

            response = generate_response(
                model,
                config,
                prompt
            )

            print(response)

        except KeyboardInterrupt:
            print("\n\nClosing LEM.")
            break

        except Exception as error:
            print("\nGeneration error:", error)


if __name__ == "__main__":
    main()
