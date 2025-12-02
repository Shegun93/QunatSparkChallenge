# Iris Classification ML Pipeline

A minimal, production-minded machine learning pipeline to classify Iris species.  
This project demonstrates a reproducible training workflow, model artifacts, and a FastAPI inference service.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#Strcuture)
- [Training](#training)  
- [Artifacts](#artifacts)  
- [Serving the Model](#serving-the-model)
- [model version mflow](#mlflow) 
- [API Usage](#api-usage)  

---

## Strcuture
```
.
├── mlruns
│   └── 0
│       └── meta.yaml
├── Problem_Desc
│   └── _Applied AI Engineer Case study-281125-120130.pdf
├── proj_image
│   ├── confusion_mflow.png
│   ├── mflow_1.png
│   ├── mflow_2.png
│   ├── modelMetrics_mflow.png
│   └── overview_mflow.png
├── requirements.txt
└── Source
    ├── artifacts
    │   ├── confusion_matrix.json
    │   ├── confusion_matrix.png
    │   ├── metrics.json
    │   ├── model.joblib
    │   └── params.json
    ├── mflow.py
    ├── mlruns
    │   ├── 0
    │   │   └── meta.yaml
    │   ├── 898090913322573235
    │   │   ├── 60b6d38b9c5841f69b884636b6571757
    │   │   │   ├── artifacts
    │   │   │   │   ├── confusion_matrix.json
    │   │   │   │   ├── confusion_matrix.png
    │   │   │   │   ├── metrics.json
    │   │   │   │   ├── model.joblib
    │   │   │   │   └── params.json
    │   │   │   ├── meta.yaml
    │   │   │   ├── metrics
    │   │   │   │   ├── accuracy
    │   │   │   │   └── f1
    │   │   │   ├── params
    │   │   │   │   ├── C
    │   │   │   │   ├── model
    │   │   │   │   ├── n_estimators
    │   │   │   │   ├── seed
    │   │   │   │   └── test_size
    │   │   │   └── tags
    │   │   │       ├── mlflow.runName
    │   │   │       ├── mlflow.source.name
    │   │   │       ├── mlflow.source.type
    │   │   │       └── mlflow.user
    │   │   └── meta.yaml
    │   └── models
    ├── processing.py
    ├── __pycache__
    │   ├── processing.cpython-39.pyc
    │   ├── serve.cpython-39.pyc
    │   └── validation.cpython-39.pyc
    ├── serve.py
    ├── train.py
    └── validation.py

16 directories, 40 files

```
## Getting Started
### Clone GitHub Repo
```
git clone https://github.com/Shegun93/QunatSparkChallenge.git
```
### Install Requirements

Install dependencies:

```
pip install -r requirements.txt
```
Dependencies include:
- Python 3.8+
- scikit-learn
- numpy
- pandas
- matplotlib
- fastapi
- uvicorn
- pydantic
- mlflow
- joblib

### Navigate to the source code folder
```
cd QunatSparkChallenge
```
## Training

Train the model using the CLI:
```
python train.py --seed 42 --test-size 0.2 --model lReg --C 1.0
```
Parameters:
- --seed: Random seed for reproducibility
- --test-size: Fraction of dataset used as test set
- --model: "lReg" (Logistic Regression) or "randF" (Random Forest)
- --C: Regularization parameter for Logistic Regression
- --n-estimators: Number of trees for Random Forest

Training will save artifacts in the artifacts/ folder and log the run in MLflow.
## Artifacts

After training, the following artifacts are saved:

| File | Description |
|------|-------------|
| `artifacts/model.joblib` | Fitted scikit-learn pipeline (scaler + model) |
| `artifacts/metrics.json` | Test set accuracy and macro F1 |
| `artifacts/params.json` | CLI parameters used during training |
| `artifacts/confusion_matrix.json` | Confusion matrix (per-class) |
| `artifacts/confusion_matrix.png` | Visual representation of confusion matrix |

## Serving the Model

Start the FastAPI server:
```
uvicorn serve:app --host 0.0.0.0 --port 8000
```
## mlflow
```
mlflow ui
```
Running that on the terminal would open the mflow ui
Real-life. It would be a remote server so that all team members can view and easily accessible for all. 
## API Usage
Health Check
```
curl http://localhost:8000/health
```
Expected Response
```
{
  "status": "ok"
}
## Prediction Example

Test the prediction endpoint with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

```
Response
```
{
  "predicted_species": "setosa",
  "class_probabilities": {
    "setosa": 0.9808,
    "versicolor": 0.0192,
    "virginica": 0.00000027
  },
  "latency_ms": 3.063,
  "model_version": "cec9b682",
  "request_id": "babfea39-4c47-4c87-b90f-3f37c3ea1319"
}

```
