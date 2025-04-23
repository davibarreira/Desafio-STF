import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tabulate import tabulate
from xgboost import XGBClassifier

# Configuration
RANDOM_STATE = 42
DATA_PATH = "data/2_pro/cleaned_dataset.parquet"


def load_and_prepare_data(
    data_path: str = DATA_PATH,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TfidfVectorizer,
    MultiLabelBinarizer,
]:
    """
    Load data and prepare it for model selection.

    Args:
        data_path: Path to the processed dataset

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb)
    """
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Prepare target variable (multi-label)
    print("Preparing data...")
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["ramo_direito"])

    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=4000)
    X = vectorizer.fit_transform(df["clean_text"])

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb


def get_model_configurations() -> list[dict]:
    """
    Define model configurations to test.

    Returns:
        List of model configurations
    """
    return [
        {
            "name": "Logistic Regression",
            "class": LogisticRegression,
            "configs": [
                {"C": 0.1, "random_state": RANDOM_STATE},
                {"C": 1.0, "random_state": RANDOM_STATE},
                {"C": 10.0, "random_state": RANDOM_STATE},
            ],
        },
        {
            "name": "Random Forest",
            "class": RandomForestClassifier,
            "configs": [
                {"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_STATE},
                {"n_estimators": 200, "max_depth": None, "random_state": RANDOM_STATE},
                {"n_estimators": 200, "max_depth": 20, "random_state": RANDOM_STATE},
            ],
        },
        {
            "name": "XGBoost",
            "class": XGBClassifier,
            "configs": [
                {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": RANDOM_STATE,
                    "objective": "binary:logistic",
                    "base_score": 0.5,
                }
            ],
        },
    ]


def train_and_evaluate_model(
    model_class: type,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[MultiOutputClassifier, float, str]:
    """
    Train and evaluate a single model configuration.

    Args:
        model_class: Classifier class
        config: Configuration parameters
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Tuple of (model, f1_score)
    """
    # Format config for display, excluding random_state
    config_str = ", ".join(f"{k}={v}" for k, v in config.items() if k != "random_state")
    print(f"\nTraining {model_class.__name__} with {config_str}")

    base_model = model_class(**config)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_val_pred, _ = multi_label_prediction(model, X_val)

    # Calculate F1 score (samples average)
    f1 = f1_score(y_val, y_val_pred, average="samples")
    print(f"F1 score (samples): {f1:.4f}")

    return model, f1, config_str


def multi_label_prediction(
    model: MultiOutputClassifier, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict multi-label values for a given feature matrix.
    For rows with no predicted labels, add the most likely label based on probabilities.

    Returns:
        Tuple of (y_pred, y_pred_proba)
    """
    y_pred_proba_list = [proba[:, 1] for proba in model.predict_proba(X)]
    y_pred_proba = np.array(y_pred_proba_list).T
    y_pred = model.predict(X)

    zero_label_rows = np.sum(y_pred, axis=1) == 0
    if np.any(zero_label_rows):
        probs_zero_rows = y_pred_proba[zero_label_rows]
        most_likely_labels = np.argmax(probs_zero_rows, axis=1)
        y_pred[zero_label_rows, most_likely_labels] = 1

    return y_pred, y_pred_proba


def evaluate_on_test_set(
    model: MultiOutputClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Evaluate the best model on the test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        F1 score on test set
    """
    # Evaluate on validation set
    y_test_pred, _ = multi_label_prediction(model, X_test)
    test_f1 = f1_score(y_test, y_test_pred, average="samples")

    print(f"\nTest Set Performance:")
    print(f"F1 Score (samples): {test_f1:.4f}")

    return test_f1


def save_best_model_config(
    best_config: dict, test_f1: float, model_dir: str = "models"
) -> None:
    """
    Save the best model configuration to disk.

    Args:
        best_config: Model configuration
        test_f1: F1 score on test set
        model_dir: Directory to save configuration
    """
    # Add performance metrics
    best_config["performance"] = {"f1_samples": float(test_f1)}

    # Create directory if it doesn't exist
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    # Save configuration
    with open(f"{model_dir}/best_model_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest model configuration saved to {model_dir}/best_model_config.json")


def save_vectorizer_and_binarizer(
    vectorizer: TfidfVectorizer, mlb: MultiLabelBinarizer, model_dir: str = "models"
) -> None:
    """
    Save the vectorizer and multilabel binarizer to disk.

    Args:
        vectorizer: Fitted TF-IDF vectorizer
        mlb: Fitted MultiLabelBinarizer
        model_dir: Directory to save artifacts
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    with open(f"{model_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"{model_dir}/multilabel_binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)


def print_comparison_table(results: list[dict]) -> None:
    """
    Print a formatted comparison table of model results.

    Args:
        results: List of model result dictionaries
    """
    table_data = [[r["model"], r["config"], f"{r['f1_samples']:.4f}"] for r in results]
    headers = ["Model", "Parameters", "F1 (Samples)"]
    print("\nModel Comparison Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def select_best_model() -> dict:
    """
    Compare different models with different parameters and select the best one
    based on f1_samples metric.
    """
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, mlb = (
        load_and_prepare_data()
    )

    # Get model configurations
    models = get_model_configurations()

    # Track results and best model
    results = []
    best_f1 = 0
    best_model = None
    best_config = None

    # Train and evaluate each model configuration
    for model_info in models:
        model_name = model_info["name"]
        model_class = model_info["class"]

        for config in model_info["configs"]:
            # Train and evaluate model
            model, f1, config_str = train_and_evaluate_model(
                model_class, config, X_train, y_train, X_val, y_val
            )

            # Track results
            results.append(
                {"model": model_name, "config": config_str, "f1_samples": f1}
            )

            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_config = {
                    "model_type": model_name,
                    "params": {k: v for k, v in config.items() if k != "random_state"},
                }

    # Display comparison table
    print_comparison_table(results)

    # Evaluate best model on test set
    print(f"\nBest model: {best_config['model_type']} with {best_config['params']}")
    test_f1 = evaluate_on_test_set(best_model, X_test, y_test)

    # Save artifacts
    save_best_model_config(best_config, test_f1)
    save_vectorizer_and_binarizer(vectorizer, mlb)

    return best_config


if __name__ == "__main__":
    select_best_model()
