"""
Lightweight hybrid model:  depthwise-separable CNN  ->  Bi-GRU over slices  ->  PoSTAL

Input  : (B, S, H, W, 1) MRI volume, values in [0, 1]
Output : dict of probabilities
           "volume" (B, 1)           tumour present in the volume?
           "slice"  (B, S, 1)        tumour present in each slice? (from the GRU)
           "mask"   (B, S, H, W, 1)  refined tumour segmentation (after PoSTAL)

Requires TensorFlow >= 2.12 (tested design targets TF 2.16+ / Keras 3).
Weights are saved/loaded with model.save_weights / load_weights, so the
architecture is rebuilt from this file (no custom-object serialisation needed).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def ds_block(filters, stride, name):
    """Depthwise-separable conv block: DW 3x3 -> BN -> ReLU6 -> PW 1x1 -> BN -> ReLU6."""
    return keras.Sequential([
        layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(max_value=6.0),
        layers.Conv2D(filters, 1, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(max_value=6.0),
    ], name=name)


class PoSTAL(layers.Layer):
    """Post-processing Spatial Attention Layer.

    Builds a spatial attention map from channel-pooled features (mean & max over
    channels) plus the coarse tumour probability, using a 7x7 conv, then uses it
    to re-weight the coarse segmentation logits. Low-attention regions (isolated,
    unsupported responses) are pushed down -> fewer false positives.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size, padding="same")
        self.shift = self.add_weight(
            name="shift", shape=(), initializer=keras.initializers.Constant(2.0), trainable=True)

    def call(self, feats, logits):
        avg = tf.reduce_mean(feats, axis=-1, keepdims=True)
        mx = tf.reduce_max(feats, axis=-1, keepdims=True)
        prob = tf.sigmoid(logits)
        att = tf.sigmoid(self.conv(tf.concat([avg, mx, prob], axis=-1)))   # (N,h,w,1)
        return logits * (0.5 + att) + self.shift * (att - 0.5)


class LightweightTumourNet(keras.Model):
    def __init__(self, slices=8, size=64, gru_units=24, **kwargs):
        super().__init__(**kwargs)
        self.S, self.H, self.W = slices, size, size
        # --- lightweight CNN (spatial features) ---
        self.stem = keras.Sequential([
            layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(), layers.ReLU(max_value=6.0)], name="stem")
        self.ds1 = ds_block(16, 2, "ds1")      # 16x16x16
        self.ds2 = ds_block(32, 2, "ds2")      # 8x8x32
        self.ds3 = ds_block(32, 1, "ds3")      # 8x8x32
        # --- GRU (inter-slice dependencies) ---
        self.gru = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True), name="bi_gru")
        self.slice_head = layers.Dense(1, name="slice_head")
        self.vol_head = keras.Sequential([layers.Dense(16, activation="relu"), layers.Dense(1)], name="vol_head")
        # --- segmentation branch + PoSTAL ---
        self.ctx_proj = layers.Dense(24, name="ctx_proj")
        self.seg_block = ds_block(24, 1, "seg_block")
        self.seg_out = layers.Conv2D(1, 1, name="seg_out")
        self.postal = PoSTAL(name="postal")

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        flat = tf.reshape(x, (-1, self.H, self.W, 1))                 # fold slices into batch
        s = self.stem(flat, training=training)
        f16 = self.ds1(s, training=training)
        f8 = self.ds3(self.ds2(f16, training=training), training=training)

        emb = tf.concat([tf.reduce_mean(f8, axis=[1, 2]), tf.reduce_max(f8, axis=[1, 2])], -1)
        emb = tf.reshape(emb, (B, self.S, 64))
        ctx = self.gru(emb, training=training)                          # (B,S,2*units)

        slice_logit = self.slice_head(ctx)                              # (B,S,1)
        vol_feat = tf.concat([tf.reduce_max(ctx, axis=1), tf.reduce_mean(ctx, axis=1)], -1)
        vol_logit = self.vol_head(vol_feat, training=training)          # (B,1)

        up = tf.image.resize(f8, (16, 16), method="bilinear")
        seg_f = self.seg_block(tf.concat([up, f16], -1), training=training)   # (N,16,16,24)
        gate = tf.reshape(self.ctx_proj(ctx), (-1, 1, 1, 24))           # inter-slice context
        seg_f = seg_f * tf.sigmoid(gate)
        coarse = self.seg_out(seg_f)
        refined = self.postal(seg_f, coarse)
        mask = tf.image.resize(refined, (self.H, self.W), method="bilinear")
        mask = tf.reshape(mask, (B, self.S, self.H, self.W, 1))

        return {"volume": tf.sigmoid(vol_logit),
                "slice": tf.sigmoid(slice_logit),
                "mask": tf.sigmoid(mask)}


def build_model(slices=8, size=64):
    m = LightweightTumourNet(slices=slices, size=size)
    m(tf.zeros((1, slices, size, size, 1)))      # build all weights
    return m


def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return bce + tf.reduce_mean(dice)


def compile_model(model, lr=2e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={"volume": "binary_crossentropy", "slice": "binary_crossentropy", "mask": bce_dice_loss},
        loss_weights={"volume": 1.0, "slice": 0.5, "mask": 1.0},
        metrics={"volume": [keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")],
                 "slice": [keras.metrics.BinaryAccuracy(name="acc")]},
    )
    return model


def load_trained(weights_path, slices=8, size=64):
    m = build_model(slices, size)
    m.load_weights(weights_path)
    return m


def predict_volume(model, vol_u8, thr=0.5):
    """vol_u8: (S,H,W) uint8 -> result dict compatible with the app's display code."""
    x = (vol_u8.astype("float32") / 255.0)[None, ..., None]
    out = model(x, training=False)
    prob = out["mask"].numpy()[0, ..., 0]
    return dict(norm=x[0, ..., 0], refined=prob, mask=prob > thr,
                slice_prob=out["slice"].numpy()[0, :, 0],
                volume_prob=float(out["volume"].numpy()[0, 0]),
                tumour_voxels=int((prob > thr).sum()), raw=None, ctx=None, brain=None)
