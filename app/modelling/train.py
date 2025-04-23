from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)

def train_model(data_path='data/2_pro/cleaned_dataset.parquet'):
    """
    Train and evaluate a multi-label classification model and save it to disk.
    
    Args:
        data_path: Path to the processed data
        model_dir: Directory to save the trained model and vectorizer
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    model_dir = Path('models')
    
    # Prepare target variable y (multi-label)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['ramo_direito'])
    
    # Create and fit vectorizer
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=4000)
    X = vectorizer.fit_transform(df['clean_text'])
    label_names = mlb.classes_
    
    # Split the data into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training model...")
    model = MultiOutputClassifier(LogisticRegression(C=1, max_iter=1000))
    # Alternative model: RandomForest
    # model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model...")
    y_pred_proba_list = [proba[:, 1] for proba in model.predict_proba(X_test)]
    # Transpose to get the correct shape (samples, labels)
    y_pred_proba = np.array(y_pred_proba_list).T
    
    # Get binary predictions
    y_pred = model.predict(X_test)
    
    # For rows with no predicted labels, add the most likely label
    zero_label_rows = np.sum(y_pred, axis=1) == 0
    if np.any(zero_label_rows):
        # Get probabilities for rows with no predictions
        probs_zero_rows = y_pred_proba[zero_label_rows]
        # Find index of highest probability label for each row
        most_likely_labels = np.argmax(probs_zero_rows, axis=1)
        # Set those labels to 1
        y_pred[zero_label_rows, most_likely_labels] = 1
    
    # Calculate evaluation metrics
    metrics = {}
    metrics['hamming_loss'] = hamming_loss(y_test, y_pred)
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    
    # Calculate percentage of cases where at least one label was correct
    at_least_one_correct = np.any((y_test == y_pred) & (y_test == 1), axis=1)
    metrics['at_least_one_correct'] = np.mean(at_least_one_correct) * 100
    
    # Calculate false positive rate
    false_positives = np.sum((y_test == 0) & (y_pred == 1))
    total_negatives = np.sum(y_test == 0)
    metrics['false_positive_rate'] = (false_positives / total_negatives) * 100
    
    # Average number of labels
    metrics['avg_labels_real'] = np.mean(np.sum(y_test, axis=1))
    metrics['avg_labels_pred'] = np.mean(np.sum(y_pred, axis=1))
    
    # Calculate PR AUC and F1 score
    metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba, average='macro')
    _, _, f1_sample, _ = precision_recall_fscore_support(y_test, y_pred, average='samples')
    metrics['f1_sample'] = f1_sample
    
    # Save the model, vectorizer, and label encoder
    print(f"Saving model and artifacts to {model_dir}")
    with open(f"{model_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(f"{model_dir}/multilabel_binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
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