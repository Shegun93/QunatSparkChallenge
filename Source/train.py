import argparse
import json
import os
import joblib
import mlflow
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from processing import load_data, split_data, preprocess
import argparse
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", type=str, choices=["lReg", "randF"], default="lReg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=100)

    return parser.parse_args()

def main():
    args = parse_args()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)


    mlflow.set_experiment("iris_classifications")
    with mlflow.start_run(run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        if args.model == "lReg":
            model = LogisticRegression(C=args.C, max_iter=200, random_state=args.seed)
        else:
            model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        pipeline = Pipeline([
            ("scaler", scaler),
            ("classifier", model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)


        joblib.dump(pipeline, os.path.join(ARTIFACT_DIR, "model.joblib"))
        with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
            json.dump({"accuracy": acc, "f1": f1}, f, indent=4)
        with open(os.path.join(ARTIFACT_DIR, "params.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        with open(os.path.join(ARTIFACT_DIR, "confusion_matrix.json"), "w") as f:
            json.dump(cm.tolist(), f, indent=4)


        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0,1,2], ["setosa","versicolor","virginica"])
        plt.yticks([0,1,2], ["setosa","versicolor","virginica"])
        plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"))
        plt.close()


        mlflow.log_params(vars(args))
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        mlflow.log_artifacts(ARTIFACT_DIR)

        print(f"\nModel trained: {args.model}")
        print(f"Hyperparameters: {vars(args)}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Macro F1: {f1:.4f}")
        print(f"Artifacts saved to {ARTIFACT_DIR}")

if __name__ == "__main__":
    main()
