# %%
"""
Realizar backtest de la estrategia
- Dado que se realiza backtest del forecast, para cada semana se tiene el
entrenamiento hasta dicha semana y luego el forecast hasta horizonte h=h
- Realizar backtest partiendo de cero inventario y ejecutar la estrategia de
inventario cada semana y obtener el costo de la estrategia (estrategia+forecast)
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

""" 2. read forecast BACKTEST - inferencia h=h para múltiples semanas """
folder_fcst_backtest = "data/submission/backtest"

data_fcst_real_train_backtest = pd.read_parquet(
    f"{folder_fcst_backtest}/data_fcst_real_train_backtest.parquet"
)
data_fcst_real_test_backtest = pd.read_parquet(
    f"{folder_fcst_backtest}/data_fcst_real_test_backtest.parquet"
)

""" 3. Definir params """
# develop = True, entrenamiento, etc
# develop = False: el productivo para generar ordenes para la semana deseada
develop = True  # backtest


""" 4. Leer state previo """
# list columns a conservar, keys + columnas necesarias para cálculos
list_info = ["unique_id", "Store", "Product"]
previous_data_state = [
    "End Inventory",
    "In Transit W+1",
    "In Transit W+2",
    "Cumulative Holding Cost",
    "Cumulative Shortage Cost",
]
previous_data_state = data_state[list_info + previous_data_state]

# si es desarrollo - simplemente se leen el orden de las filas
# se parte con inventario cero, cero tránsito w1, cero tránsito w2
if develop:
    previous_data_state.loc[:, "End Inventory"] = 0
    previous_data_state.loc[:, "In Transit W+1"] = 0
    previous_data_state.loc[:, "In Transit W+2"] = 0
    previous_data_state.loc[:, "Cumulative Holding Cost"] = 0
    previous_data_state.loc[:, "Cumulative Shortage Cost"] = 0


""" 5. Realizar for por cada ejecución del foreast y calcular orden y calcular resulados """

# obtener listado con todas las ejecuciones del backtest
list_ds_ejecuciones_fcst = (
    data_fcst_real_test_backtest["week0_update"].unique().tolist()
)

# for de cada una de las ejecuciones
for date_week0 in list_ds_ejecuciones_fcst:
    # print(date_week0)
    date_week0 = list_ds_ejecuciones_fcst[0]  # TODO: luego cambiar al for


# filtrar forecast de la fecha de ejecución filtrada
data_fcst_real_test_backtest_filtered = data_fcst_real_test_backtest[
    data_fcst_real_test_backtest["week0_update"] == date_week0
]

""" 5. Dar formato forecast generado - fechas en las columnas """
# OJO: la data FILTERED de la ejecución DESEADA
# pivotear: filas: unique_id, columnas: ds, values: forecast
values_y_fcst = "forecast_int"
data_fcst = data_fcst_real_test_backtest_filtered.pivot(
    index="unique_id", columns="ds", values=values_y_fcst
)

# renombrar columnas fcst_w1, fcst_w2, fcst_w3
data_fcst.columns = ["fcst_w1", "fcst_w2", "fcst_w3"]
data_fcst = data_fcst.reset_index()


""" 6. Ordenar fcst en el mismo orden que previus_data_state """
orden_unique_ids = previous_data_state["unique_id"].unique()

data_fcst["unique_id"] = pd.Categorical(
    data_fcst["unique_id"], categories=orden_unique_ids, ordered=True
)

data_fcst = data_fcst.sort_values("unique_id").reset_index(drop=True)

# AUX. Asegurar que state y fcst tienen el mismo orden de keys
assert (
    data_fcst["unique_id"].values == previous_data_state["unique_id"].values
).all(), (
    "¡Los unique_id no están alineados entre data_fcst y previous_data_state!"
)
