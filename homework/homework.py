#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}

import os
import pandas as pd
import gzip
import json
import pickle

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error 

def load_preprocess_data():
    train_path = 'files/input/train_data.csv.zip'
    test_path = 'files/input/test_data.csv.zip'

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    train_dataset["Age"] = 2021 - train_dataset["Year"]
    test_dataset["Age"] = 2021 - test_dataset["Year"]

    train_dataset.drop(columns=["Year", "Car_Name"], inplace=True)
    test_dataset.drop(columns=["Year", "Car_Name"], inplace=True)

    return train_dataset, test_dataset

def make_train_test_split(train_dataset, test_dataset):
    x_train = train_dataset.drop(columns=["Present_Price"])
    y_train = train_dataset["Present_Price"]
    x_test = test_dataset.drop(columns=["Present_Price"])
    y_test = test_dataset["Present_Price"]

    return x_train, y_train, x_test, y_test

def make_pipeline():
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features = ["Selling_Price", "Driven_kms", "Age", "Owner"]

    preprocessor = ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
    )

    pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('regressor', LinearRegression())
        ],
    )
    return pipeline

def make_grid_search(pipeline, x_train, y_train):
    param_grid = {
        "feature_selection__k": [5, 10, 'all']
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
    )
    grid_search.fit(x_train, y_train)

    return grid_search

def save_estimator(estimator):
    models_path = "files/models"
    os.makedirs(models_path, exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)     

def calc_metrics(model, x_train, y_train, x_test, y_test):
    metrics = []

    for x, y, label in [(x_train, y_train, 'train'), (x_test, y_test, 'test')]:
        y_pred = model.predict(x)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mad = median_absolute_error(y, y_pred)

        metrics.append({
            'type': 'metrics',
            'dataset': label,
            'r2': r2,
            'mse': mse,
            'mad': mad,
        })

    return metrics

def save_metrics(metrics):
    os.makedirs("files/output", exist_ok=True)

    with open("files/output/metrics.json", "w") as file:
        for metric in metrics:
            file.write(json.dumps(metric, ensure_ascii=False))
            file.write('\n')

def main():
    train_dataset, test_dataset = load_preprocess_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_dataset, test_dataset)
    pipeline = make_pipeline()
    model = make_grid_search(pipeline, x_train, y_train)
    save_estimator(model)
    metrics = calc_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)

if __name__ == "__main__":
    main()