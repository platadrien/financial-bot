import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on sauvegarde le PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras import callbacks, layers

load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH") or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT_PATH)

from tools.features import (  # noqa: E402
    CLIP,
    FEATURE_NAMES,
    N_FEATURES,
    N_RAW,
    RAW_COLS,
    WINDOW,
    apply_scaler,
    compute_features,
)
from tools.positionnal_encoding import PositionalEncoding  # noqa: E402

# ===================== Configuration =====================
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # le reste (15 %) = test
EPOCHS = int(os.getenv("EPOCHS", "60"))
BATCH_SIZE = 64

COST_BPS = 2.0  # coût one-way (spread + slippage) en points de base. Hors commissions !
THRESHOLD_BPS = 5.0  # seuil de signal sauvegardé pour l'API (à tuner sur la validation)
BARS_PER_DAY = 39  # barres de 10 min en séance régulière
TRADING_DAYS = 252

tf.keras.utils.set_random_seed(SEED)

# ===================== 1. Chargement des données =====================
csv_path = os.path.join(ROOT_PATH, "data", "csv", "nvda", "NVDA_10min_2years.csv")
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"], utc=True)
df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

ok = (
    (df[["open", "high", "low", "close", "average"]] > 0).all(axis=1)
    & (df["volume"] > 0)
    & (df["barCount"] > 0)
)
if (~ok).any():
    print(f"{(~ok).sum()} barres invalides supprimées")
df = df[ok].reset_index(drop=True)
print(f"{len(df)} barres — du {df['date'].iloc[0]} au {df['date'].iloc[-1]}")

# ===================== 2. Fenêtres, features et cible =====================
raw = df[RAW_COLS].to_numpy(dtype=np.float64)
n = len(raw) - N_RAW  # nombre d'échantillons (il faut une barre de plus pour la cible)

# fenêtre s : barres brutes s .. s+32 ; dernière barre de la fenêtre = e = s+32
windows = sliding_window_view(raw, N_RAW, axis=0)[:n].transpose(0, 2, 1)  # (n, 33, 7)
X_all = compute_features(windows)  # (n, 32, 7)

e = np.arange(n) + WINDOW
close = raw[:, 3]
y_all = close[e + 1] / close[e] - 1  # rendement de la barre suivante

# Cible valide seulement si la barre suivante est bien +10 min (pas de gap overnight/week-end)
next_ok = ((df["date"].shift(-1) - df["date"]) == pd.Timedelta(minutes=10)).to_numpy()
valid = next_ok[e]

# ===================== 3. Split chronologique train / val / test avec purge =====================
# GAP : évite que les fenêtres (qui se chevauchent) fuient d'un split à l'autre
GAP = N_RAW + 1
i1, i2 = int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)
idx = np.arange(n)
tr = idx[:i1][valid[:i1]]
va = idx[i1 + GAP : i2][valid[i1 + GAP : i2]]
te = idx[i2 + GAP :][valid[i2 + GAP :]]
print(f"Échantillons — train: {len(tr)}, val: {len(va)}, test: {len(te)}")

# Standardisation : paramètres calculés sur le TRAIN uniquement
flat = X_all[tr].reshape(-1, N_FEATURES)
mean = flat.mean(axis=0)
scale = np.where(flat.std(axis=0) > 1e-12, flat.std(axis=0), 1.0)
X_scaled = apply_scaler(X_all, mean, scale).astype(np.float32)

# Cible standardisée par l'écart-type du train (Huber delta=1 a alors un sens)
y_std = float(y_all[tr].std())
y_scaled = (y_all / y_std).astype(np.float32)
print(f"Écart-type du rendement 10 min (train) : {y_std * 1e4:.1f} bps")


# ===================== 4. Modèle Transformer =====================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(
        inputs, inputs
    )
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(inputs.shape[-1])(ff)  # retour à la dimension d'entrée
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


D_MODEL = 32


def build_model():
    inputs = tf.keras.Input(shape=(WINDOW, N_FEATURES))
    x = layers.Dense(D_MODEL)(inputs)  # projection des 7 features vers d_model
    x = PositionalEncoding(max_len=512, d_model=D_MODEL)(x)

    x = transformer_encoder(x, head_size=16, num_heads=4, ff_dim=64, dropout=0.1)
    x = transformer_encoder(x, head_size=16, num_heads=4, ff_dim=64, dropout=0.1)

    # Moyenne de la séquence + dernière barre (la plus récente pèse plus)
    pooled = layers.GlobalAveragePooling1D()(x)
    last = layers.Flatten()(layers.Cropping1D(cropping=(WINDOW - 1, 0))(x))
    x = layers.Concatenate()([pooled, last])

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.Huber(delta=1.0),
)
model.summary()

# ===================== 5. Entraînement (validation ≠ test) =====================
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1
)

history = model.fit(
    X_scaled[tr],
    y_scaled[tr],
    validation_data=(X_scaled[va], y_scaled[va]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ===================== 6. Évaluation sur le TEST, contre la baseline « prédis 0 » =====================
pred = (
    model.predict(X_scaled[te], verbose=0).flatten() * y_std
)  # rendement prédit (fraction)
actual = y_all[te]

mae_model = np.mean(np.abs(actual - pred)) * 1e4
mae_zero = np.mean(np.abs(actual)) * 1e4
ic = float(np.corrcoef(pred, actual)[0, 1])
rank_ic = float(pd.Series(pred).corr(pd.Series(actual), method="spearman"))
t_stat = ic * np.sqrt((len(te) - 2) / (1 - ic**2))
nz = actual != 0
hit = float(np.mean(np.sign(pred[nz]) == np.sign(actual[nz])))

print("\n================ ÉVALUATION (jeu de test) ================")
print(f"MAE modèle        : {mae_model:.2f} bps")
print(f"MAE baseline « 0 »: {mae_zero:.2f} bps   <- le modèle doit faire mieux")
print(
    f"IC (Pearson)      : {ic:+.4f}  (t={t_stat:.1f}, indicatif : les barres sont autocorrélées)"
)
print(f"IC (Spearman)     : {rank_ic:+.4f}")
print(f"Hit rate direction: {hit:.1%}")
print(
    f"Écart-type des prédictions : {pred.std() * 1e4:.2f} bps (réel : {actual.std() * 1e4:.1f} bps)"
)


def backtest(pred, actual, thr_bps, long_only=True):
    thr = thr_bps / 1e4
    if long_only:
        pos = np.where(pred > thr, 1.0, 0.0)
    else:
        pos = np.where(pred > thr, 1.0, np.where(pred < -thr, -1.0, 0.0))
    turnover = np.abs(np.diff(pos, prepend=0.0))
    pnl = (
        pos * actual - turnover * COST_BPS / 1e4
    )  # coût payé à chaque changement de position
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(BARS_PER_DAY * TRADING_DAYS)
    return pnl, pos, turnover.sum(), sharpe


bh_pnl = actual
bh_sharpe = (
    bh_pnl.mean() / (bh_pnl.std() + 1e-12) * np.sqrt(BARS_PER_DAY * TRADING_DAYS)
)
print(f"\nBacktest test, coût = {COST_BPS} bps par côté (commissions NON incluses)")
print(
    f"Buy & hold : rendement cumulé {bh_pnl.sum() * 100:+.2f} %, Sharpe {bh_sharpe:.2f}"
)
print(
    f"{'seuil':>8} {'mode':>11} {'en position':>12} {'trades':>7} {'rend. net':>10} {'Sharpe':>7}"
)
for thr in (0, 2, 5, 10, 20):
    for long_only in (True, False):
        pnl, pos, trades, sh = backtest(pred, actual, thr, long_only)
        print(
            f"{thr:>6} bp {'long-only' if long_only else 'long/short':>11} "
            f"{np.mean(pos != 0):>11.1%} {int(trades):>7} {pnl.sum() * 100:>+9.2f}% {sh:>7.2f}"
        )

# ===================== 7. Visualisation =====================
pnl_sel, _, _, _ = backtest(pred, actual, THRESHOLD_BPS, long_only=True)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(np.cumsum(bh_pnl) * 100, label="Buy & hold", alpha=0.8)
axes[0].plot(
    np.cumsum(pnl_sel) * 100,
    label=f"Stratégie long-only (seuil {THRESHOLD_BPS:g} bps, net)",
    alpha=0.8,
)
axes[0].set_title("Test — rendement cumulé (%)")
axes[0].legend()
axes[1].plot(history.history["loss"], label="Train loss")
axes[1].plot(history.history["val_loss"], label="Val loss")
axes[1].set_title("Courbe d'apprentissage")
axes[1].set_ylabel("Huber loss (cible standardisée)")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig("backtest_results.png", dpi=150)

# ===================== 8. Sauvegarde =====================
model_dir = os.path.join(ROOT_PATH, "models", "nvda")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "nvda_transformer.keras"))

meta = {
    "version": "returns-v1",
    "window": WINDOW,
    "n_raw_bars": N_RAW,
    "raw_cols": RAW_COLS,
    "feature_names": FEATURE_NAMES,
    "clip": CLIP,
    "scaler_mean": mean.tolist(),
    "scaler_scale": scale.tolist(),
    "target_std": y_std,
    "threshold_bps": THRESHOLD_BPS,
    "cost_bps_one_way": COST_BPS,
    "trained_from": str(df["date"].iloc[0]),
    "trained_to": str(df["date"].iloc[-1]),
    "test_metrics": {
        "mae_bps": float(mae_model),
        "mae_zero_bps": float(mae_zero),
        "ic": ic,
        "rank_ic": rank_ic,
        "hit_rate": hit,
    },
}
with open(os.path.join(model_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

# Prédiction sur les dernières barres (même chemin de code que l'API)
last = compute_features(raw[-N_RAW:][None])
last_pred = (
    float(
        model.predict(apply_scaler(last, mean, scale).astype(np.float32), verbose=0)[
            0, 0
        ]
    )
    * y_std
)
print(
    f"\nDernière barre : close ${raw[-1, 3]:.2f} -> rendement prévu {last_pred * 1e4:+.2f} bps "
    f"(prix prévu ${raw[-1, 3] * (1 + last_pred):.3f})"
)
print("Modèle et meta.json sauvegardés ✓  (scaler.pkl n'est plus utilisé)")
