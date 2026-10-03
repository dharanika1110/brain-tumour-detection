"""
Synthetic MRI volume generator for the brain-tumour detection project.

Each sample is a small MRI "volume" of S axial slices (default 8 x 64 x 64).
Because the project's architecture models INTER-SLICE dependencies (GRU) and
refines masks (PoSTAL), the data is generated as true 3-D structures:

  * Brain  : slice-dependent ellipse + skull ring + ventricles + smooth tissue texture
  * Tumour : 3-D ellipsoid with irregular boundary spanning several slices
             (glioma = irregular + oedema halo, meningioma = round & peripheral,
              pituitary = small & midline)
  * Distractors : small bright blobs on 1-2 slices only (vessels / artefacts).
             They are NOT tumours, so they create slice-isolated false positives
             that the GRU + PoSTAL stages are meant to suppress.
  * Noise  : Rician-style noise, level depends on simulated scanner field strength.

This is SYNTHETIC data for developing/demonstrating the pipeline.
It is NOT a substitute for BraTS / Figshare / Br35H for real evaluation.

Usage:
    python synthetic_data.py --n 500 --out data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

TUMOUR_TYPES = ["glioma", "meningioma", "pituitary"]


def _smooth_noise(rng, shape, sigma):
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-8)


def make_volume(rng, size=64, slices=8, has_tumour=True):
    """Return (volume float[0,1] (S,H,W), mask uint8 (S,H,W), info dict)."""
    S, N = slices, size
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    xx -= N / 2
    yy -= N / 2

    scanner_T = float(rng.choice([1.5, 3.0]))
    noise_sigma = 0.040 if scanner_T == 1.5 else 0.025

    a, b = N * 0.40, N * 0.45
    c_z = (S - 1) / 2
    vol = np.zeros((S, N, N), np.float32)
    mask = np.zeros((S, N, N), np.uint8)
    brain_masks = []

    # ---------- tumour parameters (3-D) ----------
    info = dict(tumour_type="none", tumour_voxels=0, tumour_slices=0,
                centre_x=-1, centre_y=-1, centre_z=-1, radius_px=0.0)
    if has_tumour:
        ttype = str(rng.choice(TUMOUR_TYPES, p=[0.45, 0.35, 0.20]))
        if ttype == "glioma":
            r = rng.uniform(6, 12); irr = 0.22
            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.10, 0.50)
        elif ttype == "meningioma":
            r = rng.uniform(5, 10); irr = 0.06
            ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.60, 0.72)
        else:  # pituitary: small, midline, slightly anterior
            r = rng.uniform(4.5, 7); irr = 0.05
            ang, rad = np.pi / 2 + rng.uniform(-0.25, 0.25), rng.uniform(0.15, 0.30)
        cx = float(np.cos(ang) * rad * a)
        cy = float(np.sin(ang) * rad * b)
        cz = float(rng.uniform(c_z - 1.5, c_z + 1.5))
        zr = float(rng.uniform(2.2, 3.6))
        core_int = float(rng.uniform(0.82, 0.93))
        info.update(tumour_type=ttype, centre_x=round(cx + N / 2, 1),
                    centre_y=round(cy + N / 2, 1), centre_z=round(cz, 1),
                    radius_px=round(r, 1))

    # ---------- distractor blobs (not tumours) ----------
    distractors = []
    if rng.random() < 0.45:
        for _ in range(int(rng.integers(1, 4))):
            distractors.append(dict(
                z=int(rng.integers(0, S)),
                x=rng.uniform(-0.45, 0.45) * a, y=rng.uniform(-0.45, 0.45) * b,
                r=rng.uniform(1.5, 3.0)))
    info["has_artefact"] = int(len(distractors) > 0)

    base_tex = _smooth_noise(rng, (S, N, N), 3.0)
    gm_wm = _smooth_noise(rng, (S, N, N), 8.0)
    bias = 1 + 0.10 * xx / N

    for z in range(S):
        sz = np.sqrt(max(0.05, 1 - ((z - c_z) / (S / 2 + 0.5)) ** 2))
        r_el = np.sqrt((xx / (a * sz)) ** 2 + (yy / (b * sz)) ** 2)
        r_el = r_el + 0.04 * _smooth_noise(rng, (N, N), 4.0)
        brain = r_el <= 1.0
        skull = (r_el > 1.0) & (r_el <= 1.12)
        img = np.zeros((N, N), np.float32)
        img[brain] = 0.42 + 0.05 * base_tex[z][brain] + 0.04 * gm_wm[z][brain]
        img[skull] = 0.75
        if sz > 0.8:  # ventricles in central slices
            for sx in (-1, 1):
                vent = ((xx - sx * 0.12 * N) / (0.05 * N * sz)) ** 2 + (yy / (0.12 * N * sz)) ** 2 <= 1
                img[vent & brain] = 0.12

        if has_tumour:
            dz = (z - cz) / zr
            if abs(dz) < 1:
                rz = r * np.sqrt(1 - dz ** 2)
                if rz >= 1.5:
                    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                    bound = rz * (1 + irr * _smooth_noise(rng, (N, N), 3.0))
                    core = (dist <= bound) & brain
                    if ttype == "glioma":
                        edema = ndi.binary_dilation(core, iterations=3) & brain & ~core
                        img[edema] = np.maximum(img[edema], 0.62)
                    img[core] = core_int + 0.03 * base_tex[z][core]
                    mask[z] = core.astype(np.uint8)

        for d in distractors:
            if d["z"] == z:
                blob = np.sqrt((xx - d["x"]) ** 2 + (yy - d["y"]) ** 2) <= d["r"]
                img[blob & brain & (mask[z] == 0)] = 0.80

        img *= bias
        vol[z] = img
        brain_masks.append(brain)

    # Rician-style noise
    n1 = rng.normal(0, noise_sigma, vol.shape)
    n2 = rng.normal(0, noise_sigma, vol.shape)
    vol = np.sqrt((vol + n1) ** 2 + n2 ** 2)
    vol = np.clip(vol, 0, 1).astype(np.float32)

    info["tumour_voxels"] = int(mask.sum())
    info["tumour_slices"] = int((mask.reshape(S, -1).sum(1) > 0).sum())
    info["scanner_T"] = scanner_T
    return vol, mask, info


def generate_dataset(n=500, size=64, slices=8, tumour_fraction=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, np.int8)
    labels[: int(round(n * tumour_fraction))] = 1
    rng.shuffle(labels)

    X = np.zeros((n, slices, size, size), np.uint8)
    M = np.zeros((n, slices, size, size), np.uint8)
    rows = []
    for i in range(n):
        v, m, info = make_volume(rng, size, slices, bool(labels[i]))
        X[i] = np.round(v * 255).astype(np.uint8)
        M[i] = m
        rows.append(dict(volume_id=f"SYN_{i:04d}", label=int(labels[i]), **info))
    meta = pd.DataFrame(rows)

    # stratified 70/15/15 split stored in the metadata so every script shares it
    split = np.empty(n, dtype=object)
    for lab in (0, 1):
        idx = rng.permutation(np.where(labels == lab)[0])
        n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    meta["split"] = split
    return X, M, labels, meta


def save_dataset(out_dir, **kw):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, M, y, meta = generate_dataset(**kw)
    np.savez_compressed(out / "synthetic_mri.npz", volumes=X, masks=M, labels=y)
    meta.to_csv(out / "metadata.csv", index=False)
    return X, M, y, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--slices", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    X, M, y, meta = save_dataset(args.out, n=args.n, size=args.size,
                                 slices=args.slices, seed=args.seed)
    print(f"Saved {len(y)} volumes {X.shape[1:]} -> {args.out}/ "
          f"(tumour volumes: {int(y.sum())})")
    print(meta.groupby(["label", "tumour_type"]).size())
