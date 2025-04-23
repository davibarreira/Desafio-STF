import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from .model_selection import multi_label_prediction


def train_model(data_path: str = "data/2_pro/cleaned_dataset.parquet") -> dict:
    """
    Train and evaluate a multi-label classification model and save it to disk.
    Uses the best model configuration from best_model_config.json and the saved vectorizer.

    Args:
        data_path: Path to the processed data

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load saved vectorizer and binarizer
    print("Loading saved vectorizer and binarizer...")
    with open(f"{model_dir}/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(f"{model_dir}/multilabel_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    # Transform text data using saved vectorizer
    print("Vectorizing text data...")
    X = vectorizer.transform(df["clean_text"])
    y = mlb.transform(df["ramo_direito"])
    label_names = mlb.classes_

    # Load best model configuration
    print("Loading best model configuration...")
    with open(f"{model_dir}/best_model_config.json", "r") as f:
        best_config = json.load(f)

    model_type = best_config["model_type"]
    params = best_config["params"]

    # Create the appropriate model class
    if model_type == "Logistic Regression":
        model_class = LogisticRegression
    elif model_type == "Random Forest":
        model_class = RandomForestClassifier
    elif model_type == "XGBoost":
        model_class = XGBClassifier
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train the model
    print(f"Training {model_type} with params: {params}")
    base_model = model_class(**params, random_state=42)
    model = MultiOutputClassifier(base_model)
    model.fit(X, y)

    # Save the model
    print(f"Saving model to {model_dir}")
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Get predictions for evaluation
    y_pred, y_pred_proba = multi_label_prediction(model, X)

    # SANITY CHECK
    # Verify that the accuracy makes sense
    # Calculate evaluation metrics
    metrics = {}
    metrics["hamming_loss"] = hamming_loss(y, y_pred)
    metrics["accuracy"] = accuracy_score(y, y_pred)

    # Calculate percentage of cases where at least one label was correct
    at_least_one_correct = np.any((y == y_pred) & (y == 1), axis=1)
    metrics["at_least_one_correct"] = np.mean(at_least_one_correct) * 100

    # Calculate false positive rate
    false_positives = np.sum((y == 0) & (y_pred == 1))
    total_negatives = np.sum(y == 0)
    metrics["false_positive_rate"] = (false_positives / total_negatives) * 100

    # Average number of labels
    metrics["avg_labels_real"] = np.mean(np.sum(y, axis=1))
    metrics["avg_labels_pred"] = np.mean(np.sum(y_pred, axis=1))

    # Calculate PR AUC and F1 score
    metrics["pr_auc"] = average_precision_score(y, y_pred_proba, average="macro")
    _, _, f1_sample, _ = precision_recall_fscore_support(y, y_pred, average="samples")
    metrics["f1_sample"] = f1_sample

    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Accuracy Score: {metrics['accuracy']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"Sample F1: {metrics['f1_sample']:.4f}")
    print(f"At least one correct: {metrics['at_least_one_correct']:.2f}%")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.2f}%")
    print(f"\nAverage labels per instance:")
    print(f"Real: {metrics['avg_labels_real']:.2f}")
    print(f"Predicted: {metrics['avg_labels_pred']:.2f}")

    return metrics


if __name__ == "__main__":
    train_model()
