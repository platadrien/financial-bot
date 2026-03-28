import logging
import os
import pickle

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ===================== Chargement modèle & scaler =====================
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/nvda_transformer.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/models/scaler.pkl")

log.info(f"Chargement du modèle : {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
