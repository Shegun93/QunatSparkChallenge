import mlflow

runs = mlflow.search_runs(experiment_names=["iris_classifications"])
print(runs)
