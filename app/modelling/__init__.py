"""
Training and prediction modules for legal text classification.
"""

from .model_selection import multi_label_prediction
from .predict import load_model, predict_labels
from .train import train_model

__all__ = ["train_model", "load_model", "predict_labels", "multi_label_prediction"]
