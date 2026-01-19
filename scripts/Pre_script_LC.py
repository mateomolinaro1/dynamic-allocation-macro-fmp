import pandas as pd
import numpy as np  
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def load_fred_md(path: str):
    """
    Charge un dataset FRED-MD :
    - extrait les codes de transformation et en fait un dico
    - supprime la ligne 'Transform:'
    - met la date en index
    """
    raw = pd.read_csv(path, header=None)

    columns = raw.iloc[0].tolist()
    transform_row = raw.iloc[1].tolist()

    # Construction du dictionnaire des codes
    transform_codes = {
        col: int(code)
        for col, code in zip(columns, transform_row)
        if col != "sasdate"
    }

    df = raw.iloc[2:].copy()
    df.columns = columns

    # Mise de la date en index
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    df = df.set_index("sasdate").sort_index()

    # Conversion en float
    df = df.apply(pd.to_numeric, errors="coerce")

    return df, transform_codes


def fred_md_transform(series: pd.Series, code: int) -> pd.Series:
    """
    Applique la transformation FRED-MD à une série.

    Parameters
    ----------
    series : pd.Series
        Série temporelle brute (index temporel).
    code : int
        Code de transformation FRED-MD (1 à 7).

    Returns
    -------
    pd.Series
        Série transformée.
    """
    s = series.astype(float)

    if code == 1:
        return s

    elif code == 2:
        return s.diff()

    elif code == 3:
        return s.diff().diff()

    elif code == 4:
        return np.log(s)

    elif code == 5:
        return np.log(s).diff()

    elif code == 6:
        return np.log(s).diff().diff()

    elif code == 7:
        return s.pct_change()

    else:
        raise ValueError("Le code de transformation doit être entre 1 et 7.")

def apply_fred_md_transformations(
    df: pd.DataFrame,
    transformation_codes: dict
) -> pd.DataFrame:
    """
    Applique les transformations FRED-MD à chaque colonne d'un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Données macro brutes.
    transformation_codes : dict
        Dictionnaire {nom_colonne: code FRED-MD}.

    Returns
    -------
    pd.DataFrame
        DataFrame transformé.
    """
    # Liste pour stocker les Series transformées
    transformed_series = []

    for col in df.columns:
        if col not in transformation_codes:
            raise KeyError(f"Code de transformation manquant pour {col}")

        code = transformation_codes[col]
        transformed_series.append(fred_md_transform(df[col], code))

    # Concaténer toutes les Series en une seule opération
    transformed = pd.concat(transformed_series, axis=1)

    return transformed

def buildLaggedFeatures(
    df_transfo: pd.DataFrame,
    lag: list
) -> pd.DataFrame:
    """
    Construit des variables décalées (lagged features) pour chaque colonne du DataFrame,
    en utilisant une liste de décalages spécifiés.

    Args:
        df_transfo (pd.DataFrame): DataFrame d'origine.
        lag (list): Liste des décalages à appliquer.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes décalées ajoutées.
    """
    new_dict = {}
    for col_name in df_transfo.columns:
        new_dict[col_name] = df_transfo[col_name]
        for l in lag:
            new_dict[f"{col_name}_lag{l}"] = df_transfo[col_name].shift(l)

    return pd.DataFrame(new_dict, index=df_transfo.index)

def handle_nans(df_transfo: pd.DataFrame, n_cols: int) -> pd.DataFrame:

    nas = [df_transfo[col].isna().sum() for col in df_transfo.columns]
    df_nas = pd.DataFrame({
        "column": df_transfo.columns,
        "nas": nas
    })
    
    df_nas = df_nas.sort_values(by="nas", ascending=False)

    return list(df_nas.head(n_cols)["column"])


### FIN DES FONCTIONS ET DEBUT DU SCRIPT PRINCIPAL ###


END_TRAIN = dt.datetime(2015, 12, 1)
START_TEST = dt.datetime(2016, 1, 1)

path = 'data/macro/FRED-MD-2024-12.csv'

#Ouverture du dataset, mise de la date en indice et récupération des codes de transfo
df, transfo_codes = load_fred_md(path)

TARGET = 'CPIAUCSL'  #Variable à prédire
TO_DROP = df.columns.difference([TARGET])  #Variables non laggées à retirer

#Application des transfo éco suivant les préconisations de la Fred
df_transformed = apply_fred_md_transformations(df, transfo_codes)

#Création des lags
df_transformed = buildLaggedFeatures(df_transformed, [1, 3, 6, 12])
#Suppression des variables non laggées
df_transformed = df_transformed.drop(TO_DROP, axis=1)


#Gestion des NA (on vire les colonnes avec le plus de NAs)
cols_high_nas = handle_nans(df_transformed, 5)
df_transformed = df_transformed.drop(cols_high_nas, axis=1)
df_transformed = df_transformed.dropna()

#Séparation des données entre variable indépendante et prédicteurs
Y = df_transformed['CPIAUCSL']
X = df_transformed.drop('CPIAUCSL', axis = 1)

assert Y.index.equals(X.index), "Les index de Y et X ne correspondent pas."

#Séparation des données entre train et test
X_train, X_test = X.loc[:END_TRAIN], X.loc[START_TEST:]
Y_train, Y_test = Y.loc[:END_TRAIN], Y.loc[START_TEST:]

#Standardisation
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

#Régularisation Lasso
tscv = TimeSeriesSplit(n_splits=5)
lasso = LassoCV(
    alphas=np.logspace(-4, 0, 50),
    max_iter=20000,
    cv=TimeSeriesSplit(5)
).fit(X_train_scaled, Y_train)

#Sélection des features
selected_features = X_train.columns[lasso.coef_ != 0]
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

#Entraînement de 2 modèle : Régression linéaire et RF

linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

linear_model.fit(X_train_selected, Y_train)
rf_model.fit(X_train_selected, Y_train)

#Evaluation des modèles
Y_pred_lin = linear_model.predict(X_test_selected)
Y_pred_rf = rf_model.predict(X_test_selected)

rmse_lin, mae_lin = np.sqrt(mse(Y_test, Y_pred_lin)), mae(Y_test, Y_pred_lin)
rmse_rf, mae_rf = np.sqrt(mse(Y_test, Y_pred_rf)), mae(Y_test, Y_pred_rf)

print(f"Linear Reg -> RMSE: {rmse_lin}, MAE: {mae_lin}")
print(f"Random Forest -> RMSE: {rmse_rf}, MAE: {mae_rf}")

plt.figure(figsize=(12, 6))
plt.plot(Y_test.index, Y_test, label='Valeurs réelles', color='blue')
plt.plot(Y_test.index, Y_pred_lin, label='Prédictions Linéaires', color='red', linestyle='--')
plt.plot(Y_test.index, Y_pred_rf, label='Prédictions Random Forest', color='green', linestyle='--')
plt.title('Prévision de l\'inflation (CPIAUCSL)')
plt.xlabel('Date')
plt.ylabel('Valeur')
plt.legend()
plt.grid()
plt.show()

#ajout