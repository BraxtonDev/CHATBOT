
"""
LEM 3.0 - PyTorch to ONNX Exporter

Converts a trained LEM checkpoint into an ONNX model
that can eventually run inside a browser using JavaScript.

Run from the CHATBOT project folder:
    python model/export_onnx.py

Install dependencies:
    python -m pip install torch onnx
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

# Make sure transformer.py can be imported
sys.path.insert(0, str(MODEL_DIR))

from transformer import LEMTransformer


class LogitsOnly(nn.Module):
    """Return only model predictions, not the training loss."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        logits, _ = self.model(input_ids)
        return logits


def load_checkpoint(path):
    """Load a PyTorch checkpoint safely."""

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

    return checkpoint


def main():

    parser = argparse.ArgumentParser(
        description="Export LEM 3.0 to ONNX"
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path"
    )

    parser.add_argument(
        "--output",
        default="browser/model/lem.onnx",
        help="ONNX output path"
    )

    args = parser.parse_args()

    # Find the checkpoint
    if args.checkpoint:
        checkpoint_path = ROOT / args.checkpoint
    else:
        chat_checkpoint = ROOT / "checkpoints/lem_chat_latest.pt"
        base_checkpoint = ROOT / "checkpoints/lem_latest.pt"

        if chat_checkpoint.exists():
            checkpoint_path = chat_checkpoint
        else:
            checkpoint_path = base_checkpoint

    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "\nNo trained checkpoint found!\n\n"
            "Expected one of:\n"
            "checkpoints/lem_chat_latest.pt\n"
            "checkpoints/lem_latest.pt\n\n"
            "Train LEM before exporting it."
        )

    print("=" * 50)
    print("LEM 3.0 - ONNX EXPORT")
    print("=" * 50)

    print(f"\nLoading: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)

    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing model weights or configuration."
        )

    config = checkpoint["config"]

    required = [
        "vocab_size",
        "block_size",
        "n_embd",
        "n_head",
        "n_layer",
        "dropout"
    ]

    for key in required:
        if key not in config:
            raise ValueError(
                f"Missing model configuration: {key}"
            )

    if int(config["vocab_size"]) != 256:
        raise ValueError(
            "This exporter expects LEM's 256-byte tokenizer."
        )

    # Reconstruct the original architecture
    print("\nRebuilding Transformer...")

    model = LEMTransformer(config)

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.eval()

    wrapper = LogitsOnly(model)
    wrapper.eval()

    # Create sample input
    block_size = int(config["block_size"])
    sample_length = min(16, block_size)

    dummy_input = torch.zeros(
        (1, sample_length),
        dtype=torch.long
    )

    # Prepare output directory
    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nExporting model to:")
    print(output_path)

    # Export model
    with torch.no_grad():

        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(output_path),

            input_names=["input_ids"],
            output_names=["logits"],

            dynamic_axes={
                "input_ids": {
                    0: "batch",
                    1: "sequence"
                },
                "logits": {
                    0: "batch",
                    1: "sequence"
                }
            },

            opset_version=17,
            do_constant_folding=True,
            dynamo=False
        )

    # Save model information
    metadata = {
        "model_name": "LEM",
        "model_version": "3.0",
        "format": "ONNX",

        "vocab_size": int(config["vocab_size"]),
        "block_size": block_size,
        "n_embd": int(config["n_embd"]),
        "n_head": int(config["n_head"]),
        "n_layer": int(config["n_layer"]),

        "checkpoint": checkpoint_path.name,
        "training_step": int(checkpoint.get("step", 0)),

        "tokenizer": "UTF-8 byte-level",
        "tokenizer_vocab_size": 256,

        "input_name": "input_ids",
        "output_name": "logits"
    }

    metadata_path = output_path.with_suffix(".json")

    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print("EXPORT COMPLETE")
    print("=" * 50)

    print(f"\nONNX model: {output_path}")
    print(f"Metadata:   {metadata_path}")

    print(f"\nTraining step: {metadata['training_step']}")
    print(f"Vocabulary:    {metadata['vocab_size']}")
    print(f"Context size:  {metadata['block_size']}")

    print("\nCreated files:")
    print("browser/model/lem.onnx")
    print("browser/model/lem.json")

    print("\nNext step: build the JavaScript inference engine.")


if __name__ == "__main__":
    main()
