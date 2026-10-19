"""
Features stationnaires, partagées entre l'entraînement et l'API.

Une seule source de vérité : si les features diffèrent entre le train et la
production, le modèle reçoit des entrées qu'il n'a jamais vues sans que rien
ne plante. Ne duplique donc jamais ce calcul ailleurs.

Entrée  : N_RAW = 33 barres brutes (32 de fenêtre + 1 barre précédente, utile
          pour calculer le rendement de la première barre de la fenêtre).
Sortie  : (32, 7) features, toutes sans unité de prix.
"""

import numpy as np

RAW_COLS = ["open", "high", "low", "close", "volume", "average", "barCount"]
WINDOW = 32
N_RAW = WINDOW + 1
FEATURE_NAMES = [
    "ret_close",  # log(close[t] / close[t-1])
    "gap_open",  # log(open[t] / close[t-1])  (inclut le gap overnight)
    "hl_range",  # log(high / low)            (volatilité de la barre)
    "clv",  # position du close dans la barre, centrée sur 0 (-0.5 .. +0.5)
    "vwap_dev",  # log(average / close)
    "vol_rel",  # log(volume / volume moyen de la fenêtre)
    "count_rel",  # log(barCount / barCount moyen de la fenêtre)
]
N_FEATURES = len(FEATURE_NAMES)
CLIP = 5.0  # écrêtage après standardisation (évite qu'un outlier écrase le modèle)


def compute_features(raw) -> np.ndarray:
    """
    raw : (..., 33, 7) barres brutes, colonnes dans l'ordre de RAW_COLS.
    Retourne (..., 32, 7) features (non standardisées).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-2:] != (N_RAW, len(RAW_COLS)):
        raise ValueError(
            f"Shape incorrecte : reçu {raw.shape[-2:]}, attendu ({N_RAW}, {len(RAW_COLS)})"
        )
    if not np.isfinite(raw).all():
        raise ValueError("NaN ou inf dans les barres")

    o, h, l, c, v, a, n = (raw[..., 1:, i] for i in range(7))
    prev_c = raw[..., :-1, 3]

    if (np.stack([o, h, l, c, a, prev_c]) <= 0).any() or (v <= 0).any() or (n <= 0).any():
        raise ValueError("Prix, volume ou barCount <= 0 dans les barres")

    rng = h - l
    clv = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1.0) - 0.5, 0.0)

    feats = np.stack(
        [
            np.log(c / prev_c),
            np.log(o / prev_c),
            np.log(h / l),
            clv,
            np.log(a / c),
            np.log(v / v.mean(axis=-1, keepdims=True)),
            np.log(n / n.mean(axis=-1, keepdims=True)),
        ],
        axis=-1,
    )
    return feats


def apply_scaler(feats, mean, scale) -> np.ndarray:
    """Standardise avec les paramètres du train, puis écrête."""
    return np.clip((feats - mean) / scale, -CLIP, CLIP)
