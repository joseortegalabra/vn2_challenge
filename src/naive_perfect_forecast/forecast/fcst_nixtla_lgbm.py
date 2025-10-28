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
from sklearn.metrics import mean_absolute_error

from utils.models_fcst import split_train_test_using_column_mark
from utils.models_fcst import train_predict_ts_mlforecast

# set root repo
set_root_path()


""" 1. read "processed" data """
# leer archivos "processed" generados en step anterior
param_index_current_week = "2"
param_value_current_date = "2024-04-22"
data, data_state, data_in_stock, data_master, data_submission = (
    read_processed_data(
        week_index=param_index_current_week,
        date_index=param_value_current_date,
    )
)


""" 2. Definir params para forecast """
horizonte_fcst = 3

# True: cuando se está haciendo pruebas/backtest/etc
# False: --> PRODUCTIVO entrenando con toda la historia disponible. Output challenge
develop = False


""" 3. Definir qué es Train y Test. Genear columna que lo identidique"""
# IMPORTANTE: SI ES PROD, EL DATA TEST, NO EXISTE POR LO QUE SE CREAN LAS FECHAS

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


""" 5. Generar features exógenas relacionadas a fechas """
# indicar de qué semana es con respecto al mes (primera semana del mes, segunda, etc)
data["week_of_month"] = data["ds"].apply(lambda d: (d.day - 1) // 7 + 1)


""" 6. Separar en data train y data test. en base a columna "TRAIN_FUTURE" """
data_train, data_test, data_test_exog = split_train_test_using_column_mark(
    df=data, verbose=True
)


""" 7. Generar modelo """
# generar modelo, entrenar y predecir TRAIN y TEST/FUTURE
data_fcst_real_train, data_fcst_real_test = train_predict_ts_mlforecast(
    df_train=data_train,
    df_test=data_test,
    df_test_exog=data_test_exog,
    horizonte_fcst=horizonte_fcst,
)

# guardar forecast (real y fcst) generados (train y test)
folder_output = "data/submission/fcst"

data_fcst_real_train.to_parquet(
    f"{folder_output}/data_fcst_real_train.parquet"
)
data_fcst_real_test.to_parquet(f"{folder_output}/data_fcst_real_test.parquet")


""" 11. Calcular métricas - SOLO APLICA PARA DEV """
# calcular MAE (con forecast decimal y con forecast int)
print("METRICS GLOBAL")

# train
mae_train = mean_absolute_error(
    y_true=data_fcst_real_train["y_true"],
    y_pred=data_fcst_real_train["forecast"],
)
mae_train_int = mean_absolute_error(
    y_true=data_fcst_real_train["y_true"],
    y_pred=data_fcst_real_train["forecast_int"],
)
print("mae_train: ", mae_train)
print("mae_train_int: ", mae_train_int)


# test - solo hay True cuando se está desarrollando
if develop:
    mae_test = mean_absolute_error(
        y_true=data_fcst_real_test["y_true"],
        y_pred=data_fcst_real_test["forecast"],
    )
    mae_test_int = mean_absolute_error(
        y_true=data_fcst_real_test["y_true"],
        y_pred=data_fcst_real_test["forecast_int"],
    )
    print("mae_test: ", mae_test)
    print("mae_test_int: ", mae_test_int)
# %%
