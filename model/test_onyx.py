
"""
LEM 3.0 - PyTorch vs ONNX Validation

Checks whether ONNX export preserved the original model.

Run:
    python model/validate_export.py

Install:
    python -m pip install torch onnxruntime numpy
"""

import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

sys.path.insert(0, str(MODEL_DIR))

from transformer import LEMTransformer


def load_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def find_checkpoint():
    chat = ROOT / "checkpoints" / "lem_chat_latest.pt"
    base = ROOT / "checkpoints" / "lem_latest.pt"

    if chat.exists():
        return chat

    if base.exists():
        return base

    raise FileNotFoundError(
        "No trained LEM checkpoint found."
    )


def main():
    print("=" * 55)
    print("LEM 3.0 - EXPORT VALIDATION")
    print("=" * 55)

    checkpoint_path = find_checkpoint()
    onnx_path = ROOT / "browser" / "model" / "lem.onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(
            "ONNX model not found. Export LEM first."
        )

    print("\nLoading PyTorch checkpoint...")
    checkpoint = load_checkpoint(checkpoint_path)

    config = checkpoint["config"]

    model = LEMTransformer(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Loading ONNX model...")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    block_size = int(config["block_size"])
    vocab_size = int(config["vocab_size"])

    test_lengths = sorted(set([
        1,
        2,
        min(16, block_size),
        block_size
    ]))

    print("\nComparing model outputs...\n")

    all_passed = True

    for length in test_lengths:
        tokens = np.random.randint(
            0,
            vocab_size,
            size=(1, length),
            dtype=np.int64
        )

        torch_input = torch.from_numpy(tokens)

        with torch.no_grad():
            torch_logits, _ = model(torch_input)

        onnx_logits = session.run(
            [output_name],
            {input_name: tokens}
        )[0]

        torch_values = torch_logits.numpy()

        difference = np.abs(
            torch_values - onnx_logits
        )

        max_difference = float(difference.max())
        mean_difference = float(difference.mean())

        passed = np.allclose(
            torch_values,
            onnx_logits,
            rtol=1e-3,
            atol=1e-4
        )

        if not passed:
            all_passed = False

        print(f"Sequence length: {length}")
        print(f"Maximum difference: {max_difference:.8f}")
        print(f"Mean difference:    {mean_difference:.8f}")
        print("Result:", "PASS" if passed else "FAIL")
        print("-" * 40)

    print("\n" + "=" * 55)

    if all_passed:
        print("SUCCESS: ONNX matches PyTorch within tolerance.")
    else:
        print("WARNING: Differences exceeded the tolerance.")
        print("Do not use this export until the issue is fixed.")

    print("=" * 55)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
