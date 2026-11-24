
# ===== 1. Importer les bibliothèques =====
import mlflow          # Module principal de MLflow
import mlflow.sklearn  # Fonctions dédiées à Scikit-learn


# ===== 2. Se connecter à la base du Model Registry (TP4) =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Charger le modèle champion depuis le Registry =====
model = mlflow.sklearn.load_model("models:/churn-model@champion")


# ===== 4. Exporter le modèle dans un dossier local =====
# serialization_format="cloudpickle" :
# Utiliser le même format de sérialisation qu'aux TP3 et TP4.
# Le dossier "model" sera ensuite copié dans l'image Docker.
# Ainsi, l'image n'aura plus besoin d'accéder au Model Registry.

mlflow.sklearn.save_model(
    model,
    "model",
    serialization_format="cloudpickle"
)


# ===== 5. Confirmer l'exportation =====
print("Champion exporté dans le dossier model/")
