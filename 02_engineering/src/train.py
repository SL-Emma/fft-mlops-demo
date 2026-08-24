import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import joblib
import argparse
import sys
import os

# Add src to path to allow running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fft_processor import generate_signals
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def run_training(n_samples=300, fs=1000, project_name="fft-mlops-demo"):
    # Generate Data
    print(f"Generating {n_samples} samples...")
    X, y, frequencies, time = generate_signals(n_samples=n_samples, fs=fs)
    
    # Setup W&B
    wandb.init(project=project_name, config={
        "n_samples": n_samples,
        "fs": fs,
        "model_type": "RandomForest"
    })

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log
    wandb.log({"accuracy": acc})
    print(f"Training Complete. Accuracy: {acc:.4f}")
    
    # Save model artifact (for deployment demo)
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    wandb.save(model_path)
    
    # Visualization (Optional, for the CI/CD logs)
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("confusion_matrix.png")
    wandb.save("confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300)
    parser_arg = parser.parse_args()
    
    run_training(n_samples=parser_arg.n_samples)
