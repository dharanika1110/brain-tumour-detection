"""
Streamlit app: Lightweight CNN + GRU + PoSTAL for real-time brain-tumour detection
on edge devices (Sethu Institute of Technology - M.E. CSE, Zeroth Review project).

Run:  streamlit run app.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import pipeline_np as P
from synthetic_data import save_dataset

st.set_page_config(page_title="Edge Brain-Tumour Detection", page_icon="🧠", layout="wide")

DATA, MODELS = Path("data"), Path("models")
WEIGHTS, METRICS = MODELS / "tumournet.weights.h5", MODELS / "metrics.json"
S_DEFAULT, SIZE = 8, 64


# ----------------------------------------------------------------------------- data / model loading
@st.cache_data(show_spinner="Generating synthetic dataset (first run only)...")
def load_data():
    if not (DATA / "synthetic_mri.npz").exists():
        save_dataset(DATA, n=500)
    d = np.load(DATA / "synthetic_mri.npz")
    return d["volumes"], d["masks"], d["labels"], pd.read_csv(DATA / "metadata.csv")


@st.cache_resource(show_spinner="Loading trained Keras model...")
def load_keras_model():
    if not WEIGHTS.exists():
        return None, "No trained weights found (run `python train.py` to create models/tumournet.weights.h5)."
    try:
        from model import load_trained
        return load_trained(str(WEIGHTS), S_DEFAULT, SIZE), None
    except Exception as e:  # TensorFlow missing / incompatible
        return None, f"Keras model could not be loaded: {e}"


def infer(vol, engine, use_gru=True, use_postal=True, open_size=5):
    if engine.startswith("Trained"):
        from model import predict_volume
        return predict_volume(KMODEL, vol)
    return P.run(vol, use_gru=use_gru, use_postal=use_postal, open_size=open_size)


# ----------------------------------------------------------------------------- plotting helpers
def montage(rows, row_titles, gt=None, final_mask=None, vmax=None):
    """rows: list of (S,H,W) arrays. Optional red contours: gt (green) / final_mask (red)."""
    S = rows[0].shape[0]
    fig, ax = plt.subplots(len(rows), S, figsize=(1.7 * S, 1.8 * len(rows)), squeeze=False)
    for r, (arr, title) in enumerate(zip(rows, row_titles)):
        for z in range(S):
            a = ax[r, z]
            a.imshow(arr[z], cmap="gray", vmin=0, vmax=vmax if vmax else arr.max() + 1e-6)
            if final_mask is not None and r == len(rows) - 1 and final_mask[z].any():
                a.contour(final_mask[z], [0.5], colors="red", linewidths=1)
            if gt is not None and r == 0 and gt[z].any():
                a.contour(gt[z], [0.5], colors="lime", linewidths=1)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"slice {z + 1}", fontsize=8)
        ax[r, 0].set_ylabel(title, fontsize=8)
    plt.tight_layout()
    return fig


def read_uploads(files, slices=S_DEFAULT, size=SIZE):
    """PNG/JPG slices (sorted by filename) or a .npy volume -> (S,H,W) uint8."""
    note = ""
    if len(files) == 1 and files[0].name.lower().endswith(".npy"):
        arr = np.load(files[0])
        arr = arr[None] if arr.ndim == 2 else arr
        arr = arr.astype(np.float32)
        arr = (255 * (arr - arr.min()) / (np.ptp(arr) + 1e-8)) if arr.max() <= 1.0 or arr.max() > 255 else arr
        sl = [np.array(Image.fromarray(a.astype(np.uint8)).resize((size, size))) for a in arr]
    else:
        files = sorted(files, key=lambda f: f.name)
        sl = [np.array(Image.open(f).convert("L").resize((size, size))) for f in files]
    sl = np.stack(sl)
    if len(sl) != slices:
        idx = np.linspace(0, len(sl) - 1, slices).round().astype(int)
        note = f"Received {len(sl)} slice(s); resampled to {slices} slices for the model."
        sl = sl[idx]
    return sl.astype(np.uint8), note


# ----------------------------------------------------------------------------- sidebar
X, M, Y, META = load_data()
KMODEL, KERR = load_keras_model()

st.sidebar.title("🧠 Edge Tumour Detector")
page = st.sidebar.radio("Navigate", ["Overview", "Dataset Explorer", "Architecture",
                                     "Detect", "Evaluation", "Edge Benchmark"])
engine_options = ["Demo engine (NumPy, no training)"]
if KMODEL is not None:
    engine_options.insert(0, "Trained Keras model (CNN+GRU+PoSTAL)")
engine = st.sidebar.selectbox("Inference engine", engine_options)
use_gru = use_postal = True
open_size = 5
if engine.startswith("Demo"):
    st.sidebar.caption("Hand-built stand-in that mirrors each pipeline stage. Not a learned model.")
    use_gru = st.sidebar.checkbox("GRU stage (inter-slice context)", True)
    use_postal = st.sidebar.checkbox("PoSTAL stage (spatial attention)", True)
    strict = st.sidebar.selectbox("CNN-stage pre-filter", ["Strict (default)", "Weak (stress test)"])
    open_size = 5 if strict.startswith("Strict") else 3
    if open_size == 3:
        st.sidebar.caption("Weak pre-filter lets more false positives through, so you can see what the GRU and "
                           "PoSTAL stages remove.")
if KMODEL is None:
    st.sidebar.info(KERR)
st.sidebar.divider()
st.sidebar.caption("Zeroth Review - Jeya Preetha. E | Guide: Dr. A. Solairaj\nSethu Institute of Technology, M.E. CSE")

# ============================================================================= OVERVIEW
if page == "Overview":
    st.title("Lightweight CNN Architecture for Real-Time Brain Tumour Detection on Edge Devices")
    st.markdown(
        "A hybrid **depthwise-separable CNN + GRU + PoSTAL** framework that detects tumours in MRI "
        "scans efficiently enough for portable, offline, point-of-care use.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Volumes in synthetic dataset", len(Y))
    c2.metric("Slices x resolution", f"{X.shape[1]} x {X.shape[2]}x{X.shape[3]}")
    c3.metric("Engine", "Keras model" if engine.startswith("Trained") else "Demo engine")

    st.subheader("Proposed pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f0fe", fontname="Helvetica"];
      A [label="MRI\\nPreprocessing"]; B [label="Lightweight CNN\\n(depthwise-separable)\\nspatial features"];
      C [label="GRU\\n(inter-slice\\ndependencies)"]; D [label="Classification"];
      E [label="PoSTAL\\n(spatial attention\\nrefinement)", fillcolor="#fde7e7"]; F [label="Final output\\nmask + verdict", fillcolor="#e6f4ea"];
      A -> B -> C -> D; C -> E -> F; D -> F; }""")

    l, r = st.columns(2)
    with l:
        st.subheader("Limitations of existing systems")
        st.markdown("- High computational cost (heavy 3D CNN / GPU)\n- Slices processed independently\n"
                    "- 3D convolutions need very large memory\n- Weak post-processing leaves false positives\n"
                    "- Large models and latency block on-device use")
    with r:
        st.subheader("What this project adds")
        st.markdown("- **CNN**: robust spatial features at low cost\n- **GRU**: models dependencies across MRI slices\n"
                    "- **PoSTAL**: refines the segmentation and cuts false positives\n"
                    "- **Edge-ready**: small footprint, low latency, no cloud needed")
    st.info("**About the data:** all scans in this app are **synthetic** (generated by `synthetic_data.py`) so the "
            "pipeline can be developed and demonstrated. For the real study, swap in BraTS / Figshare / Br35H.")

# ============================================================================= DATASET EXPLORER
elif page == "Dataset Explorer":
    st.title("Synthetic MRI Dataset")
    f1, f2, f3 = st.columns(3)
    lab = f1.selectbox("Label", ["all", "tumour", "healthy"])
    ttype = f2.selectbox("Tumour type", ["all"] + sorted(META.tumour_type.unique().tolist()))
    split = f3.selectbox("Split", ["all", "train", "val", "test"])
    df = META.copy()
    if lab != "all":
        df = df[df.label == (1 if lab == "tumour" else 0)]
    if ttype != "all":
        df = df[df.tumour_type == ttype]
    if split != "all":
        df = df[df.split == split]
    st.caption(f"{len(df)} volumes match")
    cA, cB = st.columns([2, 1])
    cA.dataframe(df)
    cB.bar_chart(META.tumour_type.value_counts())
    if len(df):
        vid = st.selectbox("Inspect volume", df.volume_id.tolist())
        i = int(META.index[META.volume_id == vid][0])
        st.write(META.loc[i].to_frame("value").T)
        fig = montage([X[i]], ["MRI"], gt=M[i], vmax=255)
        st.pyplot(fig); plt.close(fig)
        st.caption("Green contour = ground-truth tumour mask. Single-slice bright blobs without a contour are "
                   "distractors (vessels/artefacts) that the GRU + PoSTAL stages should not flag.")
    st.download_button("Download metadata.csv", META.to_csv(index=False), "metadata.csv", "text/csv")

# ============================================================================= ARCHITECTURE
elif page == "Architecture":
    st.title("Model Architecture")
    st.markdown(f"Input volume: **{S_DEFAULT} slices x {SIZE} x {SIZE}** (grayscale). Shapes below are per slice unless noted.")
    arch = pd.DataFrame([
        ("Stem", "Conv 3x3, stride 2, BN, ReLU6", "32x32x8", "Fast downsampling"),
        ("DS block 1", "Depthwise 3x3 (s2) + Pointwise 1x1", "16x16x16", "Depthwise-separable: ~8-9x fewer MACs than standard conv"),
        ("DS block 2", "Depthwise 3x3 (s2) + Pointwise 1x1", "8x8x32", "Deeper spatial features"),
        ("DS block 3", "Depthwise 3x3 + Pointwise 1x1", "8x8x32", "Feature refinement"),
        ("Slice embedding", "Global avg + max pooling", "64", "One vector per slice"),
        ("Bi-GRU", "GRU(24) forward + backward over slices", "S x 48", "Inter-slice dependencies"),
        ("Heads", "Slice head (sigmoid), Volume head (max+mean -> MLP)", "S x 1 ; 1", "Detection / classification"),
        ("Seg branch", "Upsample + skip, DS block, GRU-context gating", "16x16x24", "Context-aware segmentation"),
        ("PoSTAL", "7x7 conv on [mean-ch, max-ch, prob] -> sigmoid attention", "16x16x1", "Suppresses unsupported (false-positive) responses"),
    ], columns=["Stage", "Operation", "Output shape", "Purpose"])
    st.dataframe(arch)
    st.subheader("PoSTAL (Post-processing Spatial Attention Layer)")
    st.latex(r"A=\sigma\big(\mathrm{Conv}_{7\times7}([\mathrm{mean}_c(F),\ \mathrm{max}_c(F),\ \sigma(L)])\big)")
    st.latex(r"L_{refined}=L\cdot(0.5+A)+\beta\,(A-0.5)")
    st.caption("F = segmentation features, L = coarse logits, beta = learnable shift. Low attention pushes logits down, "
               "removing isolated responses.")
    if KMODEL is not None:
        st.subheader("Trained model")
        st.metric("Parameters", f"{KMODEL.count_params():,}")
    else:
        st.info("Train the model (`python train.py`) to see its exact parameter count here.")
    st.code(Path("model.py").read_text()[:4000], language="python")

# ============================================================================= DETECT
elif page == "Detect":
    st.title("Detect Tumour in an MRI Volume")
    src = st.radio("Input source", ["Sample from synthetic dataset", "Upload slices (PNG/JPG) or a .npy volume"],
                   horizontal=True)
    vol, gt, note, label_txt = None, None, "", ""
    if src.startswith("Sample"):
        c1, c2 = st.columns(2)
        kind = c1.selectbox("Case type", ["glioma", "meningioma", "pituitary", "healthy", "healthy with artefact blob"])
        sub = META[META.split == "test"]
        if kind == "healthy":
            sub = sub[(sub.label == 0) & (sub.has_artefact == 0)]
        elif kind.startswith("healthy with"):
            sub = sub[(sub.label == 0) & (sub.has_artefact == 1)]
        else:
            sub = sub[sub.tumour_type == kind]
        pick = c2.selectbox("Volume", sub.volume_id.tolist())
        i = int(META.index[META.volume_id == pick][0])
        vol, gt = X[i], M[i] > 0
        label_txt = f"Ground truth: **{META.tumour_type[i] if META.label[i] else 'no tumour'}**"
    else:
        files = st.file_uploader("Upload MRI slices (one PNG/JPG per slice, ordered by filename) or one .npy volume",
                                 type=["png", "jpg", "jpeg", "npy"], accept_multiple_files=True)
        if files:
            vol, note = read_uploads(files)
    if vol is not None:
        t0 = time.perf_counter()
        res = infer(vol, engine, use_gru, use_postal, open_size)
        ms = (time.perf_counter() - t0) * 1000
        found = res["volume_prob"] >= 0.5
        a, b, c, d = st.columns(4)
        (a.error if found else a.success)("TUMOUR DETECTED" if found else "NO TUMOUR DETECTED")
        b.metric("Tumour probability", f"{res['volume_prob']:.2f}")
        c.metric("Slices flagged", int(res["mask"].reshape(len(vol), -1).any(1).sum()))
        d.metric("Inference time", f"{ms:.1f} ms")
        if label_txt:
            st.markdown(label_txt)
        if note:
            st.warning(note)
        if res.get("raw") is not None:
            rows = [res["norm"], res["raw"], res["ctx"], res["refined"], res["norm"]]
            names = ["Input", "CNN stage", "+ GRU", "+ PoSTAL", "Final mask"]
            fig = montage(rows, names, gt=gt, final_mask=res["mask"], vmax=1.0)
        else:
            fig = montage([res["norm"], res["refined"], res["norm"]], ["Input", "Prob. map", "Final mask"],
                          gt=gt, final_mask=res["mask"], vmax=1.0)
        st.pyplot(fig); plt.close(fig)
        st.caption("Green = ground truth (synthetic samples only), red = predicted tumour mask. "
                   "Row-by-row you can see how each stage cleans up the previous one.")
        st.bar_chart(pd.DataFrame({"slice probability": res["slice_prob"]},
                                  index=[f"S{k + 1}" for k in range(len(vol))]))
        st.warning("Research prototype on synthetic data - not a medical device and not for clinical use.")

# ============================================================================= EVALUATION
elif page == "Evaluation":
    st.title("Evaluation")
    if engine.startswith("Trained"):
        if METRICS.exists():
            m = json.loads(METRICS.read_text())
            t = m["test"]
            cols = st.columns(5)
            for col, (k, v) in zip(cols, [("Accuracy", t["accuracy"]), ("Sensitivity", t["sensitivity"]),
                                         ("Specificity", t["specificity"]), ("AUC", t["auc"]),
                                         ("Dice (tumour vols)", t["dice_tumour_volumes"])]):
                col.metric(k, f"{v:.3f}")
            st.caption(f"Held-out test set: {t['n']} volumes | parameters: {m['params']:,} | epochs: {m['epochs_run']}")
            h = pd.DataFrame({k: v for k, v in m["history"].items() if k in ("loss", "val_loss")})
            st.line_chart(h)
        else:
            st.info("models/metrics.json not found - run `python train.py`.")
    else:
        st.warning("**Read this first:** the demo engine is a hand-built heuristic that I tuned on the same synthetic "
                   "generator it is evaluated on, so near-perfect scores are expected and say nothing about real MRI. "
                   "What *is* informative here is the ablation: what each stage (GRU, PoSTAL) removes. Switch the sidebar "
                   "pre-filter to *Weak (stress test)* to see the stages make a larger difference.")
        split = st.selectbox("Evaluate on", ["test", "val", "train", "all"])
        idx = np.arange(len(Y)) if split == "all" else np.where(META.split == split)[0]

        @st.cache_data(show_spinner="Running ablation...")
        def ablation(idx_t, osz):
            out = []
            for g, p, name in [(False, False, "CNN only"), (True, False, "CNN + GRU"),
                               (False, True, "CNN + PoSTAL"), (True, True, "CNN + GRU + PoSTAL (full)")]:
                pr, dice, fp_h = [], [], 0
                for i in idx_t:
                    r = P.run(X[i], use_gru=g, use_postal=p, open_size=osz)
                    pr.append(r["volume_prob"])
                    if Y[i] == 1:
                        gm = M[i] > 0
                        dice.append(2 * (r["mask"] & gm).sum() / (r["mask"].sum() + gm.sum() + 1e-9))
                    else:
                        fp_h += int(r["volume_prob"] >= 0.5)
                pr = np.array(pr); yt = Y[list(idx_t)]
                tn, fp, fn, tp = confusion_matrix(yt, (pr >= 0.5).astype(int), labels=[0, 1]).ravel()
                out.append(dict(config=name, accuracy=(tp + tn) / len(yt), sensitivity=tp / max(tp + fn, 1),
                                specificity=tn / max(tn + fp, 1),
                                AUC=roc_auc_score(yt, pr) if len(set(yt)) > 1 else np.nan,
                                dice=float(np.mean(dice)) if dice else np.nan,
                                healthy_vols_flagged=fp_h, fpr=pr))
            return out
        res = ablation(tuple(int(i) for i in idx), open_size)
        tbl = pd.DataFrame([{k: v for k, v in r.items() if k != "fpr"} for r in res]).set_index("config")
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in ["accuracy", "sensitivity", "specificity", "AUC", "dice"]}))
        yt = Y[idx]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        full = res[-1]
        cm = confusion_matrix(yt, (full["fpr"] >= 0.5).astype(int), labels=[0, 1])
        ax[0].imshow(cm, cmap="Blues")
        for (r_, c_), v in np.ndenumerate(cm):
            ax[0].text(c_, r_, v, ha="center", va="center", fontsize=14)
        ax[0].set_xticks([0, 1], ["healthy", "tumour"]); ax[0].set_yticks([0, 1], ["healthy", "tumour"])
        ax[0].set_xlabel("predicted"); ax[0].set_ylabel("actual"); ax[0].set_title("Confusion matrix (full)")
        for r in res:
            fpr_, tpr_, _ = roc_curve(yt, r["fpr"]); ax[1].plot(fpr_, tpr_, label=r["config"])
        ax[1].plot([0, 1], [0, 1], "k:"); ax[1].legend(fontsize=7); ax[1].set_title("ROC")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

# ============================================================================= EDGE BENCHMARK
elif page == "Edge Benchmark":
    st.title("Edge Deployment Benchmark")
    budget = st.number_input("Real-time budget per volume (ms)", 10, 2000, 100)
    n_runs = st.slider("Timed runs", 10, 200, 50)
    if st.button("Run benchmark"):
        idx = np.random.default_rng(0).choice(len(X), n_runs)
        infer(X[0], engine, use_gru, use_postal, open_size)            # warm-up
        times = []
        for i in idx:
            t0 = time.perf_counter(); infer(X[i], engine, use_gru, use_postal, open_size)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        a, b, c = st.columns(3)
        a.metric("Median latency", f"{np.median(times):.1f} ms")
        b.metric("95th percentile", f"{np.percentile(times, 95):.1f} ms")
        c.metric("Throughput", f"{1000 / np.median(times):.0f} volumes/s")
        (st.success if np.percentile(times, 95) <= budget else st.error)(
            f"95th-percentile latency {'meets' if np.percentile(times, 95) <= budget else 'exceeds'} the {budget} ms budget "
            "on THIS machine (not on an actual edge device).")
        fig, ax = plt.subplots(figsize=(6, 2.6)); ax.hist(times, bins=20); ax.axvline(budget, color="r", ls="--")
        ax.set_xlabel("ms per volume"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Model footprint")
    rows = []
    for name, path in [("Keras weights", WEIGHTS), ("TFLite model", MODELS / "tumournet.tflite")]:
        rows.append(dict(artifact=name, size_kb=round(path.stat().st_size / 1024, 1) if path.exists() else None,
                         status="found" if path.exists() else "not created yet (run train.py)"))
    st.dataframe(pd.DataFrame(rows))
    if METRICS.exists():
        m = json.loads(METRICS.read_text())
        st.caption(f"Parameters: {m['params']:,}" + (f" | TFLite median latency: {m['tflite_latency_ms_median']} ms"
                   if "tflite_latency_ms_median" in m else ""))
    st.info("Latency measured here is on your current machine. For an honest edge claim, run `models/tumournet.tflite` "
            "on the target device (e.g. Raspberry Pi / Jetson / phone) and compare against the budget.")
