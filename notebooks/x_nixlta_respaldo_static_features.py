# %%
"""
Entrenar modelo y luego realizar fcst
OJO:
- Se entrena y se hace forecast en base a la última observación del true value
- Como la competencia busca generar recomendaciones para 6-8 semanas,
    en cada ejecución semanal se entrena nuevamente el modelo y se hace forecast
- Cada semana forecast actualizado entrenando modelo con los datos más recientes
"""

import pandas as pd

from utils.utils import read_processed_data, set_root_path
from mlforecast import MLForecast
import lightgbm as lgb

from mlforecast.lag_transforms import (
    RollingMean,
    SeasonalRollingMean,
)

# set root repo
set_root_path()


""" 1. read "processed" data """
# leer archivos "processed" generados en step anterior
data, data_state, data_in_stock, data_master, data_submission = (
    read_processed_data(week_index="0", date_index="2024-04-08")
)


""" 2. Definir params para forecast """
horizonte_fcst = 2  # predicho las próximas X semanas

# si es True, es porque se está desarrollando
# y se dejan datos test para medir métricas
develop = True


""" 3. Definir qué es Train y Test. Genear columna que lo identidique"""
data_copy = data.copy()

# DEV
# separar historia en train y test. para TEST se resta el horizonte
if develop is True:
    # obtener fechas futuras - test
    # se resta de a 7 días, de esta forma cuadró con fechas en "data_in_stock"
    last_date = data["ds"].max()
    future_dates = []
    for index in range(horizonte_fcst):
        new_date = last_date - pd.Timedelta(days=7 * index)
        future_dates.append(new_date)

    # marcar las filas future. El resto es train
    data.loc[data["ds"].isin(future_dates), "TRAIN_FUTURE"] = "FUTURE"
    data["TRAIN_FUTURE"] = data["TRAIN_FUTURE"].fillna("TRAIN")


# PROD
# agregar fechas futuras que se desconoce el real
if develop is False:
    # obtener fechas futuras
    # OBS: se suma de a 7 días, de esta forma cuadró con fechas en "data_in_stock"
    last_date = data["ds"].max()
    future_dates = []
    for index in range(1, horizonte_fcst + 1):
        new_date = last_date + pd.Timedelta(days=7 * index)
        future_dates.append(new_date)
    future_dates_df = pd.DataFrame({"ds": future_dates})

    # hacer producto cartesiana de fechas futuras con "data"
    unique_ids_info = data[["unique_id", "Store", "Product"]].drop_duplicates()
    data_future = unique_ids_info.merge(future_dates_df, how="cross")
    data_future["y"] = pd.NA

    # agregar columna "TRAIN_FUTURE" para identificar el tipo de dataframe
    data["TRAIN_FUTURE"] = "TRAIN"
    data_future["TRAIN_FUTURE"] = "FUTURE"

    # hacer el merge, agregar info futura
    data = pd.concat([data, data_future], ignore_index=True)


""" 3.b Print info """
aux_data_train = data[data["TRAIN_FUTURE"] == "TRAIN"]
aux_data_test = data[data["TRAIN_FUTURE"] == "FUTURE"]
print("data shape: ", data.shape)
print("data TRAIN shape: ", aux_data_train.shape)
print("data TEST shape: ", aux_data_test.shape)

print("TRAIN min date: ", aux_data_train["ds"].min())
print("TRAIN max date: ", aux_data_train["ds"].max())

print("TEST min date: ", aux_data_test["ds"].min())
print("TEST max date: ", aux_data_test["ds"].max())

del aux_data_train
del aux_data_test


""" 4. Agregar columnas data_master. Static features """
# listado de static features que se agregarán
list_columns_data_master = data_master.columns.tolist()
list_columns_data_master = list(
    set(list_columns_data_master) - {"unique_id", "Store", "Product"}
)

# merge
data = pd.merge(
    data, data_master, on=["unique_id", "Store", "Product"], how="left"
)

# validar que no se piedan keys
list_unique_id_data_master = data_master["unique_id"].unique().tolist()
list_unique_id_data = data["unique_id"].unique().tolist()
assert set(list_unique_id_data_master) == set(
    list_unique_id_data
), "Error: los unique_id no coinciden entre data_master y data"

del data_master


""" 5. Generar features exógenas """
# indicar de qué semana es con respecto al mes (primera semana del mes, segunda, etc)
data["week_of_month"] = data["ds"].apply(lambda d: (d.day - 1) // 7 + 1)


""" 6. Separar en data train y data test """
# separar por columna auxiliar que marca qué es train y qué es future
mask_train = data["TRAIN_FUTURE"] == "TRAIN"
data_train = data[mask_train]
mask_test = data["TRAIN_FUTURE"] == "FUTURE"
data_test = data[mask_test]

# eliminar columnas auxiliares que no son fatures ni target
list_columns_to_drop = ["TRAIN_FUTURE"]
data_train = data_train.drop(columns=list_columns_to_drop)
data_test = data_test.drop(columns=list_columns_to_drop)


""" 7. Generar dataframe test - exógenas features - las necesarias para predecir """
data_test_exog = data_test.copy()

# eliminar columna y en data test. Lo que se busca predecir.
# guardar valores para comparar, solo útil en DEV
y_test_true = data_test_exog[["unique_id", "ds", "y"]].copy()
data_test_exog = data_test_exog.drop(columns="y")

# eliminar features estáticas - solo se necesitan en TRAIN
# OBS: según yo, se podrían repetir el valor como features exógenas
data_test_exog = data_test_exog.drop(columns=list_columns_data_master)


""" 8. Generar modelo """
# ejemplo M4 usando datos horarios: https://www.kaggle.com/code/lemuz90/m4-competition ejemplo M5 usando datos diario: https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval

# hiperparametros modelo LGBM - baseline y funciona bien
params_lgbm = {
    "verbose": -1,
    "num_threads": 4,
    "force_col_wise": True,
    "num_leaves": 30,
    "n_estimators": 350,
    "random_state": 42,
}

# "hiperparámetros" features lags creados por nixtla
# se prueba con un rezago de las ventas de los últimos 2 meses
max_lags = 8
list_lags = list(range(1, max_lags + 1))

# crear modelo nixtla
model_fcst = MLForecast(
    models=[lgb.LGBMRegressor(**params_lgbm)],
    freq="W-MON",  # semana comienza el lunes
    lags=list_lags,
    date_features=["month", "quarter", "week", "year"],
    lag_transforms={
        1: [
            RollingMean(4),  # media movil 1 mes
            RollingMean(8),  # media movil 2 meses
            RollingMean(12),  # media movil 3 meses
            SeasonalRollingMean(4, 4),  # estacionalidad mensual
            # ExpandingMean(),  # depende de toda la historia, mejor no usar
        ],
        4: [
            RollingMean(4),
            RollingMean(8),
            RollingMean(12),
            SeasonalRollingMean(4, 4),  # estacionalidad mensual
        ],
        8: [
            RollingMean(4),
            RollingMean(8),
            RollingMean(12),
            SeasonalRollingMean(4, 4),  # estacionalidad mensual
        ],
    },
    num_threads=-1,
)

# train
model_fcst.fit(
    data_train,
    static_features=list_columns_data_master,
    # se habilita la opción para obtener los valores forecasteado para los datos de train
    fitted=True,
)

# obs: mostrar las features usadas por el modelo
# model_fcst.ts.features_order_

""" 9. Predecir con el modelo """
predictions = model_fcst.predict(
    h=horizonte_fcst,
    X_df=data_test_exog,  # exógenas necesarias para predecir
    # new_df=data_train, # necesario cuando se entrena el modelo y se guarda y se hace inferencia con otros datos
)


# por qué se perdió algo
debugging = model_fcst.get_missing_future(
    h=horizonte_fcst, X_df=data_test_exog
)


# %%
