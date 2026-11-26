
"""
LEM 3.0 - ONNX Model Tester

Run:
    python model/test_onnx.py

Install:
    python -m pip install onnxruntime numpy
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "browser" / "model" / "lem.onnx"
CONFIG_PATH = ROOT / "browser" / "model" / "lem.json"


def main():
    print("=" * 50)
    print("LEM 3.0 - ONNX MODEL TEST")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run python model/export_onnx.py first."
        )

    print("\nLoading model...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print("Input:", input_info.name)
    print("Output:", output_info.name)

    if CONFIG_PATH.exists():
        config = json.loads(
            CONFIG_PATH.read_text(encoding="utf-8")
        )
        block_size = int(config["block_size"])
        vocab_size = int(config["vocab_size"])
    else:
        block_size = 128
        vocab_size = 256

    print("\nTesting predictions...")

    test_lengths = sorted(set([
        1,
        2,
        min(16, block_size),
        block_size
    ]))

    for length in test_lengths:
        tokens = np.random.randint(
            0,
            vocab_size,
            size=(1, length),
            dtype=np.int64
        )

        results = session.run(
            [output_info.name],
            {input_info.name: tokens}
        )

        logits = results[0]

        expected_shape = (1, length, vocab_size)

        if logits.shape != expected_shape:
            raise RuntimeError(
                f"Incorrect output shape: {logits.shape}; "
                f"expected {expected_shape}"
            )

        if not np.isfinite(logits).all():
            raise RuntimeError(
                "Model produced NaN or infinite values."
            )

        next_token = int(
            np.argmax(logits[0, -1, :])
        )

        if not 0 <= next_token < vocab_size:
            raise RuntimeError("Invalid predicted token.")

        print(
            f"PASS: input length {length}, "
            f"output shape {logits.shape}, "
            f"predicted token {next_token}"
        )

    print("\n" + "=" * 50)
    print("ALL ONNX TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
