import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from app.modelling.model_selection import (
    evaluate_on_test_set,
    get_model_configurations,
    load_and_prepare_data,
    print_comparison_table,
    train_and_evaluate_model,
)

# Load sample data
SAMPLE_DATA_PATH = "tests/sample_data/sample_data.parquet"


def test_load_and_prepare_data():
    """Test data loading and preparation."""

    # Test the actual function
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb = (
        load_and_prepare_data(SAMPLE_DATA_PATH)
    )
    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_val.shape[0] > 0
    assert y_test.shape[0] > 0
    assert isinstance(vectorizer, object)
    assert isinstance(mlb, object)


def test_get_model_configurations():
    """Test model configurations generation."""
    configs = get_model_configurations()
    assert len(configs) > 0
    assert all("name" in config for config in configs)
    assert all("class" in config for config in configs)
    assert all("configs" in config for config in configs)


def test_train_and_evaluate_model():
    """Test model training and evaluation."""
    # Load and prepare sample data
    df = pd.read_parquet(SAMPLE_DATA_PATH)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["ramo_direito"])

    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=4000)
    X = vectorizer.fit_transform(df["texto_bruto"])

    # Create a simple config
    config = {"C": 1.0, "random_state": 42}

    # Train and evaluate - the function returns model, f1, config_str
    model, f1, config_str = train_and_evaluate_model(
        LogisticRegression, config, X, y, X, y
    )
    results = [{"model": "LogisticRegression", "config": config_str, "f1_samples": f1}]

    assert model is not None
    assert 0 <= f1 <= 1
    assert isinstance(config_str, str)

    assert 0 <= evaluate_on_test_set(model, X, y) <= 1
    assert print_comparison_table(results) is None


# def test_evaluate_on_test_set():
#     """Test test set evaluation."""
#     X_test = np.array([[1, 2], [3, 4]])
#     y_test = np.array([[1, 0], [0, 1]])

#     config = {
#         'model': LogisticRegression,
#         'params': {'C': 1.0}
#     }

#     model, _ = train_and_evaluate_model(config['model'], config, X_test, y_test, X_test, y_test)
#     f1 = evaluate_on_test_set(model, X_test, y_test)
#     assert 0 <= f1 <= 1
