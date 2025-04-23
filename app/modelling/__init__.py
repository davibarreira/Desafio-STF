"""
Training and prediction modules for legal text classification.
"""

from .predict import load_model, predict_labels
from .train import train_model

__all__ = ["train_model", "load_model", "predict_labels"]
