#!/usr/bin/env python3
"""Build projection/residual controls from a CAA vector and feature directions."""

import argparse
from pathlib import Path

import torch

from grounding_utils import normalize


def load_vec(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        obj = next(v for v in obj.values() if torch.is_tensor(v))
    return torch.as_tensor(obj, dtype=torch.float32).flatten()


def orthonormal_basis(vecs):
    basis = []
    for vec in vecs:
        v = vec.float().clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        if v.norm() > 1e-8:
            basis.append(normalize(v))
    return basis


def project(vec, basis):
    out = torch.zeros_like(vec)
    for b in basis:
        out = out + torch.dot(vec, b) * b
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caa-vector", required=True)
    parser.add_argument("--feature-vectors", required=True, help="comma-separated .pt paths")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="caa_sae1246_200_62")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    caa = normalize(load_vec(args.caa_vector))
    features = [load_vec(p.strip()) for p in args.feature_vectors.split(",") if p.strip()]
    basis = orthonormal_basis(features)
    proj = normalize(project(caa, basis))
    resid = normalize(caa - project(caa, basis))
    rand = normalize(torch.randn(caa.shape, generator=torch.Generator().manual_seed(args.seed)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(proj, out / f"{args.prefix}_proj_L17.pt")
    torch.save(resid, out / f"{args.prefix}_resid_L17.pt")
    torch.save(-caa, out / f"{args.prefix}_reverse_L17.pt")
    torch.save(rand, out / f"{args.prefix}_random_L17.pt")
    print({
        "basis_size": len(basis),
        "proj_cos": float(torch.dot(caa, proj)),
        "resid_cos": float(torch.dot(caa, resid)),
        "out_dir": str(out),
    })


if __name__ == "__main__":
    main()
