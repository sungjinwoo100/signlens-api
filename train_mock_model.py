"""
Generates and trains a baseline scikit-learn RandomForestClassifier on 63 landmark features
for ASL letters A-Z and exports the trained model to ml/asl_model.pkl.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_and_save_model(output_path: str = "ml/asl_model.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    samples_per_label = 20
    n_features = 63  # 21 landmarks x 3 axes (x, y, z)

    X_list = []
    y_list = []

    np.random.seed(42)
    for idx, label in enumerate(labels):
        # Generate distinct synthetic landmark clusters for each letter
        base_center = np.random.uniform(0.1, 0.9, size=n_features)
        for _ in range(samples_per_label):
            noise = np.random.normal(0, 0.05, size=n_features)
            X_list.append(base_center + noise)
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    print(f"Training RandomForestClassifier on {len(X)} synthetic landmark samples across {len(labels)} classes...")
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)

    joblib.dump(model, output_path)
    print(f"Successfully trained and saved model to '{output_path}'.")


if __name__ == "__main__":
    train_and_save_model()
