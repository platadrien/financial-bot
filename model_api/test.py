import logging
import os
import pickle

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras import callbacks, layers


class PositionalEncoding(layers.Layer):
    """
    Positional encoding sinusoïdal compatible avec n'importe quelle
    dimension d_model (pair ou impair).
    Précalculé à l'init pour éviter les problèmes de shape dynamique.
    """

    def __init__(self, max_len=512, d_model=7, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Précalcul numpy → shape (1, max_len, d_model)
        import numpy as np

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        positions = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        dims = np.arange(d_model)[np.newaxis, :]  # (1, d_model)
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        pe[:, 0::2] = np.sin(angles[:, 0::2])  # colonnes paires  → sin
        pe[:, 1::2] = np.cos(angles[:, 1::2])  # colonnes impaires → cos
        # Stocké comme poids non-entraînable
        self.pe = tf.constant(pe[np.newaxis, :, :])  # (1, max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)


# ===================== Chargement modèle & scaler =====================
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/nvda_transformer.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/models/scaler.pkl")

log.info(f"Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
    },
)
