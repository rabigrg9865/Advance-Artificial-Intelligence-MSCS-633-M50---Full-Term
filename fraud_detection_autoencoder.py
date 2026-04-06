"""
Fraud Detection
Name: Rabi gurung
Assignment: Hands-On Assignment 4: Use Unsupervised Deep Learning Algorithm to Detect Fraud with PyOD
Course: MSCS-633-M50 (Advanced Artificial Intelligence)

This application detects fraud using an AutoEncoder model.
"""

from __future__ import annotations

# Import required libraries
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Configuration class to store all experiment settings
@dataclass
class ExperimentConfig:
    data_path: str = "data/creditcard.csv"  # path to dataset
    test_size: float = 0.20  # 20% data for testing
    random_state: int = 42  # for reproducibility
    contamination_floor: float = 0.001  # minimum fraud ratio
    hidden_neuron_list: tuple[int, ...] = (32, 16, 8)  # neural network layers
    epoch_num: int = 18  # training iterations
    batch_size: int = 1024  # batch size
    lr: float = 0.001  # learning rate
    dropout_rate: float = 0.10  # dropout to prevent overfitting
    batch_norm: bool = True  # normalize layers
    optimizer_name: str = "adam"  # optimizer
    verbose: int = 2  # training output level
    train_on_normal_only: bool = True  # train only on non-fraud data


# Create required folders (data, outputs, etc.)
def ensure_directories() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    directories = {
        "project_root": project_root,
        "data": project_root / "data",
        "outputs": project_root / "outputs",
        "references": project_root / "references",
    }
    # Create folders if they don’t exist
    for directory in directories.values():
        if isinstance(directory, Path):
            directory.mkdir(parents=True, exist_ok=True)
    return directories


# Load dataset from CSV file
def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    # Check if target column exists
    if "Class" not in df.columns:
        raise ValueError("Dataset must include a 'Class' column.")
    return df


# Prepare data (split, scale, and filter)
def prepare_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    train_on_normal_only: bool,
) -> dict[str, object]:

    # Separate features and labels
    feature_frame = df.drop(columns=["Class"]).copy()
    labels = df["Class"].astype(int).copy()

    # Split into training and testing sets
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        feature_frame,
        labels,
        test_size=test_size,
        stratify=labels,  # keep fraud ratio same
        random_state=random_state,
    )

    # Normalize data (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # Train only on normal data if selected
    if train_on_normal_only:
        normal_mask = y_train.to_numpy() == 0
        X_train_model = X_train_scaled[normal_mask]
    else:
        X_train_model = X_train_scaled

    # Calculate fraud percentage
    train_contamination = float(y_train.mean())

    # Return processed data and stats
    return {
        "feature_names": feature_frame.columns.tolist(),
        "scaler": scaler,
        "X_train_model": X_train_model,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "train_contamination": train_contamination,
        "dataset_profile": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "fraud_count": int(labels.sum()),
            "fraud_rate_percent": round(float(labels.mean()) * 100, 4),
            "train_rows": int(len(X_train_df)),
            "test_rows": int(len(X_test_df)),
            "train_fraud_count": int(y_train.sum()),
            "test_fraud_count": int(y_test.sum()),
            "model_training_rows": int(len(X_train_model)),
        },
    }


# Train AutoEncoder model
def train_model(X_train_model: np.ndarray, contamination: float, config: ExperimentConfig) -> dict[str, object]:

    # Ensure contamination is not too small
    effective_contamination = max(contamination, config.contamination_floor)

    # Create AutoEncoder model
    detector = AutoEncoder(
        contamination=effective_contamination,
        preprocessing=False,
        lr=config.lr,
        epoch_num=config.epoch_num,
        batch_size=config.batch_size,
        optimizer_name=config.optimizer_name,
        random_state=config.random_state,
        verbose=config.verbose,
        hidden_neuron_list=list(config.hidden_neuron_list),
        batch_norm=config.batch_norm,
        dropout_rate=config.dropout_rate,
    )

    # Measure training time
    started = time.perf_counter()
    detector.fit(X_train_model)
    elapsed_seconds = time.perf_counter() - started

    return {
        "model": detector,
        "train_seconds": round(elapsed_seconds, 2),
        "effective_contamination": effective_contamination,
    }


# Evaluate model performance
def evaluate_model(model: AutoEncoder, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:

    # Get anomaly scores (higher = more likely fraud)
    anomaly_scores = model.decision_function(X_test)

    # Predict labels (0 = normal, 1 = fraud)
    predicted_labels = model.predict(X_test).astype(int)

    # Calculate metrics
    precision = precision_score(y_test, predicted_labels, zero_division=0)
    recall = recall_score(y_test, predicted_labels, zero_division=0)

    report = classification_report(
        y_test,
        predicted_labels,
        target_names=["Normal", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix values
    matrix = confusion_matrix(y_test, predicted_labels)
    tn, fp, fn, tp = matrix.ravel()

    # Store all results
    metrics = {
        "threshold": float(model.threshold_),
        "roc_auc": float(roc_auc_score(y_test, anomaly_scores)),
        "average_precision": float(average_precision_score(y_test, anomaly_scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(report["Fraud"]["f1-score"]),
        "mcc": float(matthews_corrcoef(y_test, predicted_labels)),
        "accuracy": float(report["accuracy"]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_predictions": int(predicted_labels.sum()),
    }

    return {
        "metrics": metrics,
        "classification_report": report,
        "anomaly_scores": anomaly_scores,
        "predicted_labels": predicted_labels,
        "confusion_matrix": matrix,
    }


# Save prediction results to CSV
def save_predictions(output_dir: Path, y_test, anomaly_scores, predicted_labels) -> None:
    predictions = pd.DataFrame(
        {
            "actual_class": y_test.astype(int),
            "predicted_class": predicted_labels.astype(int),
            "anomaly_score": anomaly_scores,
        }
    ).sort_values(by="anomaly_score", ascending=False)

    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


# Save metrics in JSON format
def save_metrics_json(output_dir: Path, config, dataset_profile, training_details, evaluation_details) -> None:

    payload = {
        "config": {**asdict(config), "hidden_neuron_list": list(config.hidden_neuron_list)},
        "dataset_profile": dataset_profile,
        "training_details": {
            "train_seconds": training_details["train_seconds"],
            "effective_contamination": training_details["effective_contamination"],
        },
        "metrics": evaluation_details["metrics"],
        "classification_report": evaluation_details["classification_report"],
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# Main function to run the full pipeline
def run_experiment(config: ExperimentConfig) -> dict[str, object]:

    # Setup folders
    directories = ensure_directories()

    # Load dataset
    data_path = directories["project_root"] / config.data_path
    df = load_dataset(str(data_path))

    # Prepare data
    prepared = prepare_data(
        df=df,
        test_size=config.test_size,
        random_state=config.random_state,
        train_on_normal_only=config.train_on_normal_only,
    )

    # Train model
    training_details = train_model(
        X_train_model=prepared["X_train_model"],
        contamination=prepared["train_contamination"],
        config=config,
    )

    # Evaluate model
    evaluation_details = evaluate_model(
        model=training_details["model"],
        X_test=prepared["X_test"],
        y_test=prepared["y_test"],
    )

    # Save results
    output_dir = directories["outputs"]

    save_predictions(
        output_dir,
        prepared["y_test"],
        evaluation_details["anomaly_scores"],
        evaluation_details["predicted_labels"],
    )

    save_metrics_json(
        output_dir,
        config,
        prepared["dataset_profile"],
        training_details,
        evaluation_details,
    )

    return {
        "dataset_profile": prepared["dataset_profile"],
        "training_details": training_details,
        "evaluation_details": evaluation_details,
        "config": config,
    }