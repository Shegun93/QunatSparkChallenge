import time
import hashlib
import uuid
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from validation import IrisFeatures


MODEL_PATH = "artifacts/model.joblib"
model = joblib.load(MODEL_PATH)


with open(MODEL_PATH, "rb") as f:
    model_hash = hashlib.sha256(f.read()).hexdigest()[:8]
SPECIES = ["setosa", "versicolor", "virginica"]

app = FastAPI(
    title="Iris Prediction API",
    description="Predicts Iris species",
    version="1.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures, request: Request):
    start_time = time.time()
    req_id = str(uuid.uuid4())

    try:
        X = np.array([[features.sepal_length,
                       features.sepal_width,
                       features.petal_length,
                       features.petal_width]])
        pred_index = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        latency_ms = round((time.time() - start_time) * 1000, 3)

        class_probs = {SPECIES[i]: float(probs[i]) for i in range(len(SPECIES))}

    
        print(f"[Request ID: {req_id}] Latency: {latency_ms} ms, Predicted: {SPECIES[pred_index]}")

        return {
            "predicted_species": SPECIES[pred_index],
            "class_probabilities": class_probs,
            "latency_ms": latency_ms,
            "model_version": model_hash,
            "request_id": req_id
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")