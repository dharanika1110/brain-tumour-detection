# Edge Brain-Tumour Detection - CNN + GRU + PoSTAL (Streamlit)

Project: *Lightweight CNN Architecture for Real-Time Brain Tumour Detection on Edge Devices*

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app works immediately with a **Demo engine** (NumPy only, no TensorFlow needed).
The synthetic dataset is in `data/` (it is regenerated automatically if missing).

## Train the real model (needs TensorFlow >= 2.16)
```bash
pip install -r requirements-train.txt          # adds TensorFlow (local machine only)
python synthetic_data.py --n 500 --out data   # optional: regenerate / enlarge dataset
python train.py --epochs 30                   # writes models/tumournet.weights.h5, metrics.json, tumournet.tflite
streamlit run app.py                          # sidebar now offers "Trained Keras model"
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: Overview, Dataset Explorer, Architecture, Detect, Evaluation, Edge Benchmark |
| `synthetic_data.py` | Synthetic 3-D MRI volume generator (tumours, distractors, noise) |
| `pipeline_np.py` | Demo engine: non-learned stand-in mirroring CNN -> GRU -> PoSTAL stages |
| `model.py` | Keras model: depthwise-separable CNN, Bi-GRU over slices, PoSTAL layer |
| `train.py` | Training, test evaluation, TFLite export + latency |
| `data/synthetic_mri.npz` | volumes (N,8,64,64) uint8, masks, labels |
| `data/metadata.csv` | per-volume info + train/val/test split |

## Moving to real data (BraTS / Figshare / Br35H)
Prepare arrays in the same layout (`volumes` (N,S,H,W) uint8, `masks`, `labels`) and a `metadata.csv`
with `volume_id,label,tumour_type,split,...` and point `--data` at that folder.

## Deploying on Streamlit Community Cloud
Use `requirements.txt` (no TensorFlow) and set the main file path to `app.py`. The app runs in Demo mode there.

Research prototype only - not a medical device.
