
import pandas as pd
train_demo = pd.read_csv("traindemographics.csv")
train_perf = pd.read_csv("trainperf.csv")
train_prev = pd.read_csv("trainprevloans.csv")



import pandas as pd

dem = pd.read_csv("traindemographics.csv")
perf = pd.read_csv("trainperf.csv")
prev = pd.read_csv("trainprevloans.csv")

print("DEMOGRAPHICS :", dem.columns.tolist())
print("PERF :", perf.columns.tolist())
print("PREVIOUS LOANS :", prev.columns.tolist())


df = dem.merge(perf, on="customerid", how="left")
df = df.merge(prev, on="customerid", how="left")

# Agrégation de l'historique des prêts
prev_agg = prev.groupby("customerid").agg({
    "systemloanid": "count",         # nombre de prêts précédents
    "loanamount": ["sum", "mean", "max"],
    "totaldue": ["sum", "mean"],
    "termdays": "mean"
}).reset_index()

# Renommer les colonnes pour plus de clarté
prev_agg.columns = [
    "customerid",
    "prev_count",
    "prev_loanamount_sum",
    "prev_loanamount_mean",
    "prev_loanamount_max",
    "prev_totaldue_sum",
    "prev_totaldue_mean",
    "prev_termdays_mean"
]

prev_agg.head()

# 1) Fusion demographics + perf
df = perf.merge(dem, on="customerid", how="left")

# 2) Fusion avec historique agrégé
df_final = df.merge(prev_agg, on="customerid", how="left")

# Vérification
print("Taille finale :", df_final.shape)
df_final

cols_to_drop = ["customerid", "systemloanid", "loannumber", "approveddate",
                "creationdate", "referredby"]

df_final = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns])

def clean_and_encode(df, date_cols=['birthdate'], target_col='good_bad_flag'):
    df = df.copy()

    # Retirer la cible
    y = df[target_col].map({'Good':0, 'Bad':1})
    df = df.drop(columns=[target_col])

    # Supprimer dates
    df = df.drop(columns=[col for col in date_cols if col in df.columns])

    # Colonnes numériques
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0)

    # Colonnes catégorielles
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[obj_cols] = df[obj_cols].fillna('Unknown')

    # OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_encoded = ohe.fit_transform(df[obj_cols])
    ohe_cols = ohe.get_feature_names_out(obj_cols)

    df_ohe = pd.DataFrame(ohe_encoded, index=df.index, columns=ohe_cols)

    df = df.drop(columns=obj_cols)
    df_clean = pd.concat([df, df_ohe], axis=1)

    return df_clean, y, ohe, obj_cols

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df_clean, y, ohe, obj_cols = clean_and_encode(df_final)

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1) Séparer la cible
# -------------------------------
X = df_clean
y = df_final["good_bad_flag"].map({'Good': 0, 'Bad': 1})

y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Avant SMOTE :", X_train.shape, y_train.value_counts())
print("Après SMOTE :", X_resampled.shape, y_resampled.value_counts())

# Remplacer tous les caractères non alphanumériques par un underscore
X_resampled.columns = [c.replace(' ', '_').replace('-', '_').replace('/', '_') for c in X_resampled.columns]

# Ou plus robuste : garder uniquement lettres, chiffres et underscore
import re
X_resampled.columns = [re.sub(r'\W+', '_', c) for c in X_resampled.columns]

import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Séparer en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

# Grille de recherche aléatoire
param_dist = {
    'num_leaves': sp_randint(20, 50),
    'learning_rate': sp_uniform(0.01, 0.1),
    'n_estimators': sp_randint(100, 1000),
    'subsample': sp_uniform(0.6, 0.4),
    'colsample_bytree': sp_uniform(0.6, 0.4),
    'min_child_samples': sp_randint(10, 50)
}

# Recherche aléatoire avec 20 combinaisons
random_search = RandomizedSearchCV(
    lgb_model, param_distributions=param_dist,
    n_iter=20, scoring='f1', cv=3, verbose=2, random_state=42
)

# Entraîner
random_search.fit(X_train, y_train)

# Meilleur modèle
best_model = random_search.best_estimator_
print("Meilleurs paramètres :", random_search.best_params_)

# Prédictions
y_pred = best_model.predict(X_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Meilleurs paramètres
best_params = {
    'colsample_bytree': 0.6727299868828402,
    'learning_rate': 0.02834045098534338,
    'min_child_samples': 21,
    'n_estimators': 413,
    'num_leaves': 41,
    'subsample': 0.6028265220878869
}

# Création du modèle final
final_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    **best_params
)

# Entraînement
final_model.fit(X_resampled, y_resampled)

# Prédictions
y_pred = final_model.predict(X_test)

# Évaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))