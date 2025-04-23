"""
Training and prediction modules for legal text classification.
"""

from .train import train_model
from .predict import load_model, predict_labels

__all__ = ["train_model", "load_model", "predict_labels"] 