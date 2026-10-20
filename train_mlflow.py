
# ===== 1. Importer les bibliothèques =====
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===== 2. Charger les données =====
df = pd.read_csv("churn_data.csv")

# Variable cible : 1 = le client part, 0 = le client reste
y = df["churn"]


# ===== 3. Préparer les données selon les colonnes choisies =====
def preparer(colonnes):

    # Convertir les variables catégorielles en colonnes numériques
    X = pd.get_dummies(
        df[colonnes],
        drop_first=True
    )

    # Diviser les données : 80 % entraînement et 20 % test
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ===== 4. Exécuter un essai et enregistrer les résultats =====
def lancer_essai(nom, n_estimators, max_depth, colonnes):

    # Préparer les données pour cet essai
    X_train, X_test, y_train, y_test = preparer(colonnes)

    # Démarrer une nouvelle exécution MLflow
    with mlflow.start_run(run_name=nom):

        # Créer le modèle Random Forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédire les résultats sur les données de test
        pred = model.predict(X_test)

        # Enregistrer les paramètres du modèle
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # Calculer les métriques de performance
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test, pred, zero_division=0
        )
        recall = recall_score(
            y_test, pred, zero_division=0
        )
        f1 = f1_score(
            y_test, pred, zero_division=0
        )

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Sauvegarder le modèle entraîné
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Afficher les résultats de l'essai
        print(
            nom,
            "-> accuracy :", round(accuracy, 3),
            "| f1 :", round(f1, 3)
        )


# ===== 5. Lancer les expériences MLflow =====

# Définir l'expérience
mlflow.set_experiment("churn-prediction")

# Liste complète des variables explicatives
TOUT = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation"
]

# Essai 1 : modèle initial, 100 arbres et profondeur 5
lancer_essai(
    "essai-1",
    100,
    5,
    TOUT
)

# Essai 2 : modèle plus simple, 50 arbres et profondeur 3
lancer_essai(
    "essai-2",
    50,
    3,
    TOUT
)

# Essai 3 : même configuration que l'essai 1,
# mais sans la variable salaire
lancer_essai(
    "essai-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"]
)
