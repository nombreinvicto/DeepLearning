import os
import optuna
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
import dask.array as da
from sklearn.metrics import accuracy_score

def main():
    # Connect to Dask cluster (workers will be provisioned by Vertex AI)
    client = Client()  # Works locally, or connects to cluster if multiple workers
    print("Connected to Dask cluster:", client)

    # Generate synthetic dataset (replace with your own data loader)
    X, y = da.make_classification(
        n_samples=200000, n_features=50, chunks=10000, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    # Objective function for Optuna
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 500)

        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid).compute()
        y_true = y_valid.compute()
        acc = accuracy_score(y_true, y_pred)
        return acc

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)

    # Save best params
    os.makedirs("model", exist_ok=True)
    with open("model/best_params.txt", "w") as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    main()
