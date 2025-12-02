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
### Requirements

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
## Serving the Model

Start the FastAPI server:
```uvicorn serve:app --host 0.0.0.0 --port 8000
```
