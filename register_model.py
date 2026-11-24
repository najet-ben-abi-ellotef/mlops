
# ===== 1. Importer les bibliothèques =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Connexion au serveur MLflow Docker =====
mlflow.set_tracking_uri("http://localhost:5000")

# ===== 3. Retrouver le meilleur run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

best = runs.iloc[0]
run_id = best["run_id"]

print("Meilleur run :", best["tags.mlflow.runName"])
print("Run ID :", run_id)

# ===== 4. Enregistrer le modèle dans le Registry =====
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Modèle enregistré :", result.version)

# ===== 5. Attribuer l'alias champion =====
client = MlflowClient()

client.set_registered_model_alias(
    "churn-model",
    "champion",
    result.version
)

print("Alias champion -> version", result.version)
