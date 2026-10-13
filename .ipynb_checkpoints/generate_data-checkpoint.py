import pandas as pd
import numpy as np


# ============================================================
# 1. Fixer la graine aléatoire
# ============================================================

# La seed permet d'obtenir les mêmes données aléatoires
# à chaque exécution du programme.
np.random.seed(42)


# ============================================================
# 2. Définir le nombre de clients
# ============================================================

n = 500  # Nombre de clients fictifs à générer


# ============================================================
# 3. Générer les données des clients
# ============================================================

# On construit le tableau colonne par colonne.
# Chaque colonne représente une FEATURE (donnée d'entrée
# du futur modèle de Machine Learning).

data = pd.DataFrame({

    # Âge du client : entre 18 et 69 ans
    "age": np.random.randint(18, 70, n),

    # Salaire mensuel : entre 1200 et 7999
    "salaire": np.random.randint(1200, 8000, n),

    # Type d'abonnement
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"],
        n
    ),

    # Ancienneté du client : entre 0 et 14 ans
    "anciennete": np.random.randint(0, 15, n),

    # Niveau de consommation du service : entre 0 et 99
    "consommation": np.random.randint(0, 100, n)
})


# ============================================================
# 4. Créer la TARGET : churn
# ============================================================

# La colonne "churn" représente ce que notre futur modèle
# devra prédire.
#
# Règle simplifiée :
# - consommation < 30
# ET
# - ancienneté < 2 ans
#
# Dans ce cas, le client est considéré comme étant
# à risque de départ (churn = 1).
#
# Sinon : churn = 0.

data["churn"] = (
    (data["consommation"] < 30)
    & (data["anciennete"] < 2)
).astype(int)


# ============================================================
# 5. Sauvegarder les données dans un fichier CSV
# ============================================================

# index=False permet de ne pas ajouter l'index de Pandas
# comme colonne supplémentaire dans le fichier CSV.

data.to_csv("churn_data.csv", index=False)


# ============================================================
# 6. Vérifier le résultat
# ============================================================

# Afficher les 5 premières lignes
print(data.head())


# Afficher le taux de churn
print(f"Taux de churn : {data['churn'].mean():.1%}")