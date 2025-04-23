import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def load_model(model_dir: Union[str, Path] = "models") -> Dict:
    """
    Load the trained model and related artifacts for prediction.

    Args:
        model_dir: Directory where model artifacts are stored

    Returns:
        Dictionary containing the model, vectorizer and label binarizer
    """
    model_dir = Path(model_dir)

    # Load the vectorizer
    with open(model_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load the multilabel binarizer
    with open(model_dir / "multilabel_binarizer.pkl", "rb") as f:
        mlb = pickle.load(f)

    # Load the model
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    return {"model": model, "vectorizer": vectorizer, "mlb": mlb}


def predict_labels(
    texts: List[str], model_dir: Union[str, Path] = "models"
) -> Union[List[str], List[List[str]]]:
    """
    Predict the legal branch labels for the given texts.

    Args:
        texts: List of texts to predict labels for
        model_dir: Directory where model artifacts are stored (used if model_artifacts is None)

    Returns:
        Dictionary containing predicted labels and probabilities
    """
    # Load model artifacts if not provided
    model_artifacts = load_model(model_dir)

    model = model_artifacts["model"]
    vectorizer = model_artifacts["vectorizer"]
    mlb = model_artifacts["mlb"]

    # Vectorize the input texts
    X = vectorizer.transform(texts)

    # Get predicted probabilities
    y_pred_proba_list = [proba[:, 1] for proba in model.predict_proba(X)]
    y_pred_proba = np.array(y_pred_proba_list).T

    # Get binary predictions
    y_pred = model.predict(X)

    # For rows with no predicted labels, add the most likely label
    zero_label_rows = np.sum(y_pred, axis=1) == 0
    if np.any(zero_label_rows):
        # Get probabilities for rows with no predictions
        probs_zero_rows = y_pred_proba[zero_label_rows]
        # Find index of highest probability label for each row
        most_likely_labels = np.argmax(probs_zero_rows, axis=1)
        # Set those labels to 1
        for i, row_idx in enumerate(np.where(zero_label_rows)[0]):
            y_pred[row_idx, most_likely_labels[i]] = 1

    # Convert binary predictions to label lists
    predicted_labels = mlb.inverse_transform(y_pred)

    # Convert tuples to lists
    predicted_labels = [list(labels) for labels in predicted_labels]
    if len(predicted_labels) == 1:
        predicted_labels = list(predicted_labels[0])

    return predicted_labels
