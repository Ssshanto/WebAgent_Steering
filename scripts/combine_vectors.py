#!/usr/bin/env python3
"""Create a normalized weighted sum of cached steering vectors."""

import argparse
from pathlib import Path

import torch

from grounding_utils import normalize


def load_vec(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        tensors = [v for v in obj.values() if torch.is_tensor(v)]
        if len(tensors) != 1:
            raise ValueError(f"{path} contains {len(tensors)} tensors; pass single-layer .pt files")
        obj = tensors[0]
    return torch.as_tensor(obj, dtype=torch.float32).flatten()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, help="Comma-separated .pt vectors")
    parser.add_argument("--weights", default=None, help="Optional comma-separated weights")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = [p.strip() for p in args.vectors.split(",") if p.strip()]
    if not paths:
        parser.error("--vectors must contain at least one path")
    weights = (
        [float(x.strip()) for x in args.weights.split(",") if x.strip()]
        if args.weights
        else [1.0] * len(paths)
    )
    if len(weights) != len(paths):
        parser.error("--weights length must match --vectors length")

    total = None
    for path, weight in zip(paths, weights):
        vec = normalize(load_vec(path))
        total = weight * vec if total is None else total + weight * vec
    out_vec = normalize(total)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_vec.cpu(), out)
    print({"vectors": len(paths), "out": str(out), "norm": float(out_vec.norm())})


if __name__ == "__main__":
    main()
