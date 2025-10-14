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


""" 4. Generar features exógenas """
pass

print(data.shape)
data.head()
# %%
