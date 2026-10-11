"""Expand a 48-D RSL-RL checkpoint with zero-initialized height-scan inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def expand_columns(tensor: torch.Tensor, target_width: int) -> torch.Tensor:
    """Append zero-valued input columns while preserving existing weights."""
    if tensor.ndim != 2 or tensor.shape[1] >= target_width:
        raise ValueError(f"Expected a 2-D tensor narrower than {target_width}, got {tuple(tensor.shape)}")
    expanded = tensor.new_zeros((tensor.shape[0], target_width))
    expanded[:, : tensor.shape[1]] = tensor
    return expanded


def expand_normalizer(tensor: torch.Tensor, target_width: int, fill_value: float) -> torch.Tensor:
    """Append neutral running-statistic values for new observations."""
    if tensor.ndim != 2 or tensor.shape[1] >= target_width:
        raise ValueError(f"Expected a 2-D tensor narrower than {target_width}, got {tuple(tensor.shape)}")
    expanded = tensor.new_full((tensor.shape[0], target_width), fill_value)
    expanded[:, : tensor.shape[1]] = tensor
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original 48-D checkpoint.")
    parser.add_argument("output", type=Path, help="Destination for the expanded checkpoint.")
    parser.add_argument("--target-width", type=int, default=235, help="Expanded observation width.")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]

    for network in ("actor", "critic"):
        key = f"{network}.0.weight"
        state[key] = expand_columns(state[key], args.target_width)

        prefix = f"{network}_obs_normalizer"
        state[f"{prefix}._mean"] = expand_normalizer(state[f"{prefix}._mean"], args.target_width, 0.0)
        state[f"{prefix}._var"] = expand_normalizer(state[f"{prefix}._var"], args.target_width, 1.0)
        state[f"{prefix}._std"] = expand_normalizer(state[f"{prefix}._std"], args.target_width, 1.0)

    # Start a new fine-tuning run and deliberately do not reuse optimizer
    # moments whose tensors correspond to the old first-layer shape.
    checkpoint["iter"] = 0
    checkpoint["infos"] = {**(checkpoint.get("infos") or {}), "height_transfer": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Expanded checkpoint written to: {args.output}")


if __name__ == "__main__":
    main()
