"""LSTM Autoencoder compatibility shim."""
from ..drift_detection import (
    SentimentWindowDataset,
    collate_fn,
    LSTMAutoencoder,
    masked_mse_loss,
    train_model,
    score_anomalies,
    detect_lstm_ae_product,
)
