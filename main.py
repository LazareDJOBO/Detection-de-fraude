# main.py
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import ast

app = Flask(__name__)

# ---------------------------
# 1) Définir les colonnes du modèle
# ---------------------------
# Colles ici exactement la liste que tu as fournie (ou charge depuis joblib)
model_columns = [
'loanamount','totaldue','termdays','longitude_gps','latitude_gps',
'prev_count','prev_loanamount_sum','prev_loanamount_mean',
'prev_loanamount_max','prev_totaldue_sum','prev_totaldue_mean',
'prev_termdays_mean','bank_account_type_Current',
'bank_account_type_Other','bank_account_type_Savings',
'bank_account_type_Unknown','bank_name_clients_Access Bank',
'bank_name_clients_Diamond Bank','bank_name_clients_EcoBank',
'bank_name_clients_FCMB','bank_name_clients_Fidelity Bank',
'bank_name_clients_First Bank','bank_name_clients_GT Bank',
'bank_name_clients_Heritage Bank','bank_name_clients_Keystone Bank',
'bank_name_clients_Skye Bank','bank_name_clients_Stanbic IBTC',
'bank_name_clients_Standard Chartered',
'bank_name_clients_Sterling Bank','bank_name_clients_UBA',
'bank_name_clients_Union Bank','bank_name_clients_Unity Bank',
'bank_name_clients_Unknown','bank_name_clients_Wema Bank',
'bank_name_clients_Zenith Bank',
'bank_branch_clients_ IDI - ORO MUSHIN',
'bank_branch_clients_17, SANUSI FAFUNWA STREET, VICTORIA ISLAND, LAGOS',
'bank_branch_clients_40,SAPELE ROAD ,OPPOSITE DUMAZ JUNCTION BENIN CITY EDO STATE.',
'bank_branch_clients_ABEOKUTA','bank_branch_clients_ABULE EGBA',
'bank_branch_clients_ACCESS BANK PLC, CHALLENGE ROUNDABOUT IBADAN, OYO STATE.',
'bank_branch_clients_ADEOLA HOPEWELL',
'bank_branch_clients_AJOSE ADEOGUN','bank_branch_clients_AKURE BRANCH',
'bank_branch_clients_AKUTE','bank_branch_clients_APAPA',
'bank_branch_clients_BOSSO ROAD, MINNA',
'bank_branch_clients_DUGBE,IBADAN','bank_branch_clients_GBAGADA',
'bank_branch_clients_LAGOS','bank_branch_clients_LEKKI EPE',
'bank_branch_clients_MAFOLUKU','bank_branch_clients_MUSHIN BRANCH',
'bank_branch_clients_OAU ILE IFE','bank_branch_clients_OBA ADEBIMPE',
'bank_branch_clients_OBA AKRAN',
'bank_branch_clients_OBA AKRAN BERGER PAINT',
'bank_branch_clients_OGBA','bank_branch_clients_OGUDU, OJOTA',
'bank_branch_clients_OJUELEGBA',
'bank_branch_clients_PLOT 999C DANMOLE STREET, ADEOLA ODEKU, VICTORIA ISLAND, LAGOS',
'bank_branch_clients_RING ROAD',
'bank_branch_clients_STERLING BANK PLC 102, IJU ROAD, IFAKO BRANCH',
'bank_branch_clients_TINCAN','bank_branch_clients_TRANS AMADI',
'bank_branch_clients_Unknown','bank_branch_clients_WHARF ROAD, APAPA',
'employment_status_clients_Contract',
'employment_status_clients_Permanent',
'employment_status_clients_Retired',
'employment_status_clients_Self-Employed',
'employment_status_clients_Student',
'employment_status_clients_Unemployed',
'employment_status_clients_Unknown',
'level_of_education_clients_Graduate',
'level_of_education_clients_Post-Graduate',
'level_of_education_clients_Primary',
'level_of_education_clients_Secondary',
'level_of_education_clients_Unknown'
]

# ---------------------------
# 2) Indiquer explicitement les colonnes numériques (les 12 premières dans ta liste)
# ---------------------------
numeric_cols = [
'loanamount','totaldue','termdays','longitude_gps','latitude_gps',
'prev_count','prev_loanamount_sum','prev_loanamount_mean',
'prev_loanamount_max','prev_totaldue_sum','prev_totaldue_mean',
'prev_termdays_mean'
]

# ---------------------------
# 3) Extraire les bases catégorielles et leurs valeurs (options)
#    On regroupe les colonnes OHE par prefixe (tout avant le dernier underscore)
# ---------------------------
def build_categorical_options(model_cols, numeric_cols):
    cat_cols = [c for c in model_cols if c not in numeric_cols]
    groups = {}
    for col in cat_cols:
        # prefix = part avant la dernière underscore
        if '_' in col:
            prefix, suffix = col.rsplit('_', 1)
        else:
            prefix, suffix = col, ''
        # strip possible trailing/leading spaces on suffix
        suffix = suffix.strip()
        groups.setdefault(prefix, []).append(suffix)
    return groups

cat_options = build_categorical_options(model_columns, numeric_cols)
# cat_options is a dict: { "bank_account_type": ["Current","Other",...], ... }

# ---------------------------
# 4) Charger le modèle
# ---------------------------
model = joblib.load("credit_final_model.pkl")  # Assure-toi que ce fichier est présent

# ---------------------------
# 5) Routes
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    # On envoie numeric_cols et cat_options au template pour générer le formulaire
    return render_template("index.html", numeric_cols=numeric_cols, cat_options=cat_options)

@app.route("/predict", methods=["POST"])
def predict_fraud():
    try:
        # 1) Récupérer valeurs numériques
        input_data = {}
        for nc in numeric_cols:
            raw = request.form.get(nc, "")
            if raw == "":
                val = 0.0
            else:
                try:
                    val = float(raw)
                except:
                    val = 0.0
            input_data[nc] = val

        # 2) Récupérer choix catégoriels (le template fournira le selected string)
        selected = {}
        for prefix in cat_options.keys():
            # name in form is the prefix (on remplace espaces par __ pour sécurité)
            form_name = prefix
            val = request.form.get(form_name, None)
            if val is None:
                val = "Unknown"
            selected[prefix] = val

        # 3) Construire DataFrame one-hot aligned with model_columns
        row = {}
        # remplir numériques
        for nc in numeric_cols:
            row[nc] = input_data[nc]

        # initialiser toutes les OHE colonnes à 0
        for col in model_columns:
            if col not in numeric_cols:
                row[col] = 0

        # activer les colonnes correspondant aux choix de l'utilisateur
        for prefix, choice in selected.items():
            # la colonne attendue est prefix + '_' + choice
            # mais attention aux espaces/ponctuation : on utilise la même string (choice peut contenir virgules)
            target_col = f"{prefix}_{choice}"
            # si exact match
            if target_col in row:
                row[target_col] = 1
            else:
                # parfois le suffix contient des espaces, vérifions une correspondance "loose"
                # on cherche dans model_columns la colonne commençant par prefix + '_'
                # et dont suffix contient choice (case-insensitive)
                found = False
                for col in model_columns:
                    if col.startswith(prefix + "_"):
                        # suffix
                        suffix = col[len(prefix)+1:]
                        if choice.lower() == suffix.lower():
                            row[col] = 1
                            found = True
                            break
                # else on laisse tout à 0 (option inconnue -> treated as none)
                if not found:
                    pass

        # 4) Construire DataFrame final
        df_input = pd.DataFrame([row], columns=model_columns)

        # 5) Prédiction
        y_pred = model.predict(df_input)[0]
        # si le modèle n'a pas predict_proba, on attrape l'exception
        try:
            y_prob = model.predict_proba(df_input)[0][1]
        except:
            y_prob = None

        label = "Fraude (Bad)" if int(y_pred) == 1 else "Normal (Good)"
        prob = round(float(y_prob), 3) if y_prob is not None else None

        return render_template("index.html",
                               numeric_cols=numeric_cols,
                               cat_options=cat_options,
                               prediction=label,
                               probability=prob)

    except Exception as e:
        return render_template("index.html",
                               numeric_cols=numeric_cols,
                               cat_options=cat_options,
                               error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
