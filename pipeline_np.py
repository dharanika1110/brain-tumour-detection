"""
NumPy "demo engine" that mirrors the stages of the proposed architecture:

   MRI preprocessing -> feature/candidate map (stand-in for the lightweight CNN)
   -> inter-slice context (stand-in for the GRU) -> PoSTAL-style spatial attention
   -> classification

IMPORTANT: this is a hand-built, non-learned stand-in so the Streamlit app runs
without TensorFlow. The real learned model lives in model.py / train.py.
"""
import numpy as np
from scipy import ndimage as ndi


def preprocess(vol_u8):
    """Normalise to [0,1] and build a skull-stripped brain mask per slice."""
    v = vol_u8.astype(np.float32)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    brain = np.zeros(v.shape, bool)
    for z in range(v.shape[0]):
        m = ndi.gaussian_filter(v[z], 1.0) > 0.12
        m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=1))
        lab, n = ndi.label(m)
        if n:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
        brain[z] = ndi.binary_erosion(m, iterations=3)  # removes bright skull ring
    return v, brain


def cnn_stage(v, brain, open_size=5):
    """Stand-in for CNN feature extraction -> per-slice tumour-likelihood map.
    Uses a volume-level healthy-tissue reference so large tumours don't bias it."""
    p = np.zeros_like(v)
    if brain.sum() < 50:
        return p
    ref = np.median(v[brain])
    for z in range(v.shape[0]):
        if brain[z].sum() < 50:
            continue
        s = ndi.gaussian_filter(v[z], 0.8)
        cand = 1 / (1 + np.exp(-(s - ref - 0.25) / 0.04))
        cand = np.where(brain[z], cand, 0)
        # thin bright rims (skull remnants) are removed by a small opening
        keep = ndi.binary_opening(cand > 0.5, structure=np.ones((open_size, open_size)))
        p[z] = np.where(ndi.binary_dilation(keep, iterations=2), cand, 0)
    return p.astype(np.float32)


def gru_stage(p, gate=0.5):
    """Stand-in for the GRU: bidirectional gated recurrence over slices.
    A real tumour has overlapping evidence in neighbouring slices; an isolated
    single-slice blob (artefact / vessel) does not and gets down-weighted."""
    S = p.shape[0]
    fwd, bwd = np.zeros_like(p), np.zeros_like(p)
    for t in range(1, S):
        fwd[t] = gate * fwd[t - 1] + (1 - gate) * p[t - 1]
    for t in range(S - 2, -1, -1):
        bwd[t] = gate * bwd[t + 1] + (1 - gate) * p[t + 1]
    ctx = np.maximum(fwd, bwd)
    ctx = np.stack([ndi.maximum_filter(c, size=7) for c in ctx])  # spatial tolerance
    ctx = np.clip(ctx / 0.5, 0, 1)
    return (p * (0.15 + 0.85 * ctx)).astype(np.float32)


def postal_stage(p, thr=0.5, min_voxels=12, min_slices=2):
    """PoSTAL-style post-processing spatial attention: a 7x7 attention map built
    from local average + local max evidence (like channel-pooled spatial
    attention), applied to the map, followed by removal of tiny / single-slice
    components to cut false positives."""
    att = np.zeros_like(p)
    for z in range(p.shape[0]):
        avg = ndi.uniform_filter(p[z], size=7)
        mx = ndi.maximum_filter(p[z], size=7)
        att[z] = 1 / (1 + np.exp(-8 * (0.5 * avg + 0.5 * mx - 0.45)))
    refined = p * att
    mask = refined > thr
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    for i in range(1, n + 1):
        comp = lab == i
        n_sl = int((comp.reshape(comp.shape[0], -1).sum(1) > 0).sum())
        if comp.sum() < min_voxels or n_sl < min_slices:
            mask[comp] = False
    return refined.astype(np.float32), mask


def run(vol_u8, use_gru=True, use_postal=True, open_size=5):
    """Full pipeline. Returns dict with intermediate maps for visualisation."""
    v, brain = preprocess(vol_u8)
    raw = cnn_stage(v, brain, open_size)
    ctx = gru_stage(raw) if use_gru else raw
    if use_postal:
        refined, mask = postal_stage(ctx)
    else:
        refined, mask = ctx, ctx > 0.5
    slice_prob = np.array([float(refined[z].max()) if mask[z].any() else float(refined[z].max()) * 0.5
                           for z in range(v.shape[0])])
    n_vox = int(mask.sum())
    lab, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    largest = int(max(ndi.sum(mask, lab, range(1, n + 1)))) if n else 0
    vol_prob = float(1 - np.exp(-largest / 40.0))
    return dict(norm=v, brain=brain, raw=raw, ctx=ctx, refined=refined, mask=mask,
                slice_prob=slice_prob, volume_prob=vol_prob, tumour_voxels=n_vox)
