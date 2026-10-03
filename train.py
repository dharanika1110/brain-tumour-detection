"""
Train the CNN + GRU + PoSTAL model on the synthetic dataset (or swap in BraTS slices
prepared in the same (N, S, H, W) layout).

    python synthetic_data.py --n 500 --out data      # once
    python train.py --epochs 30

Outputs (in ./models): tumournet.weights.h5, metrics.json, tumournet.tflite (if conversion works)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
from model import build_model, compile_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out", default="models")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
tf.keras.utils.set_random_seed(0)

d = np.load(f"{args.data}/synthetic_mri.npz")
meta = pd.read_csv(f"{args.data}/metadata.csv")
X, M, y = d["volumes"], d["masks"], d["labels"].astype("float32")
N, S, H, W = X.shape


def pack(idx, flip=False):
    x = X[idx].astype("float32")[..., None] / 255.0
    m = M[idx].astype("float32")[..., None]
    if flip:  # horizontal-flip augmentation
        x, m = x[:, :, :, ::-1], m[:, :, :, ::-1]
    s = (m.reshape(len(idx), S, -1).max(-1, keepdims=True)).astype("float32")
    t = {"volume": y[idx][:, None], "slice": s, "mask": m}
    return np.ascontiguousarray(x), {k: np.ascontiguousarray(v) for k, v in t.items()}


tr = np.where(meta.split == "train")[0]
va = np.where(meta.split == "val")[0]
te = np.where(meta.split == "test")[0]
(x_a, t_a), (x_b, t_b) = pack(tr), pack(tr, flip=True)
x_tr = np.concatenate([x_a, x_b])
t_tr = {k: np.concatenate([t_a[k], t_b[k]]) for k in t_a}
x_va, t_va = pack(va)

model = compile_model(build_model(S, H))
print("Parameters:", model.count_params())
t0 = time.time()
hist = model.fit(x_tr, t_tr, validation_data=(x_va, t_va), epochs=args.epochs,
                 batch_size=args.batch, shuffle=True, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                            restore_best_weights=True)])
train_time = time.time() - t0

# ---------- test evaluation ----------
x_te, t_te = pack(te)
pred = model.predict(x_te, batch_size=32, verbose=0)
pv = pred["volume"][:, 0]
yt = y[te]
tn, fp, fn, tp = confusion_matrix(yt, (pv >= 0.5).astype(int), labels=[0, 1]).ravel()
pm = pred["mask"][..., 0] > 0.5
gm = M[te] > 0
dice = [2 * (pm[i] & gm[i]).sum() / (pm[i].sum() + gm[i].sum() + 1e-9) for i in range(len(te)) if yt[i] == 1]

weights_path = f"{args.out}/tumournet.weights.h5"
model.save_weights(weights_path)
metrics = dict(
    params=int(model.count_params()), epochs_run=len(hist.history["loss"]), train_seconds=round(train_time, 1),
    test=dict(n=int(len(te)), accuracy=float((tp + tn) / len(te)), sensitivity=float(tp / max(tp + fn, 1)),
              specificity=float(tn / max(tn + fp, 1)), auc=float(roc_auc_score(yt, pv)),
              dice_tumour_volumes=float(np.mean(dice)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    weights_kb=round(os.path.getsize(weights_path) / 1024, 1),
    history={k: [float(v) for v in vals] for k, vals in hist.history.items()},
)

# ---------- TFLite export for edge deployment (best effort) ----------
try:
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    blob = conv.convert()
    open(f"{args.out}/tumournet.tflite", "wb").write(blob)
    metrics["tflite_kb"] = round(len(blob) / 1024, 1)
    itp = tf.lite.Interpreter(model_content=blob); itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    xi = x_te[:1].astype(inp["dtype"]); ts = []
    for _ in range(20):
        t = time.perf_counter(); itp.set_tensor(inp["index"], xi); itp.invoke(); ts.append(time.perf_counter() - t)
    metrics["tflite_latency_ms_median"] = round(float(np.median(ts) * 1000), 2)
except Exception as e:  # noqa
    metrics["tflite_error"] = str(e)[:300]
    print("TFLite export failed:", e)

json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
