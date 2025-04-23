import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from app.modelling.train import train_model

# Load sample data
SAMPLE_DATA_PATH = "tests/sample_data/sample_data.parquet"


def test_train_model():
    metrics = train_model(SAMPLE_DATA_PATH, save_model=False)
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert all(
        key in metrics
        for key in [
            "hamming_loss",
            "accuracy",
            "at_least_one_correct",
            "false_positive_rate",
            "avg_labels_real",
            "avg_labels_pred",
            "pr_auc",
            "f1_sample",
        ]
    )
