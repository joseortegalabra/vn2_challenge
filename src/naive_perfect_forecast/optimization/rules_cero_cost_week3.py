# %%
"""
Naive Optimization
Basada en forecast y reglas
- Como se predice la venta de las próximas 2 semanas, se puede calcular
el stock al final de la semana 1 y al conocer el forecast de demanda de la
semana 2, se puede saber cuánto pedir.
- Aplicando alguna regla para sobre-estimar ya que el costo de subestimar es mayor
al de sobreestimar
- Lo más importante es tener un forecast lo más preciso posible
"""

import pandas as pd
import numpy as np

from utils.utils import read_processed_data, set_root_path


# set root repo
set_root_path()


""" 1. read "processed" data """
# leer archivos "processed" generados en step anterior
data, data_state, data_in_stock, data_master, data_submission = (
    read_processed_data(week_index="0", date_index="2024-04-08")
)


""" 2. read forecast generados """
folder_fcst = "data/submission/fcst"

data_fcst_output_train = pd.read_parquet(
    f"{folder_fcst}/data_fcst_output_train.parquet"
)
data_fcst_output_test = pd.read_parquet(
    f"{folder_fcst}/data_fcst_output_test.parquet"
)

""" 3. Definir params """
# develop = True, entrenamiento, etc
# develop = False: el productivo para generar ordenes para la semana deseada
develop = False


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


""" 5. Dar formato forecast generado - fechas en las columnas """
# pivotear: filas: unique_id, columnas: ds, values: forecast
values_y_fcst = "LGBMRegressor_int"
data_fcst = data_fcst_output_test.pivot(
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


""" 6. generar reglas - asume forecast perfectos y objetivo costo CERO en W+3"""
# crear variables con los nombres de acuerdo a la fórmula calculada
initial_inventory_w1 = (
    previous_data_state["End Inventory"]
    + previous_data_state["In Transit W+1"]
)

fcst_w1 = data_fcst["fcst_w1"]
fcst_w2 = data_fcst["fcst_w2"]
fcst_w3 = data_fcst["fcst_w3"]

inventory_transit_w2 = previous_data_state["In Transit W+2"]

min_initial_inventory_w1_fcst_w1 = -np.minimum(initial_inventory_w1, fcst_w1)


###### generar regla calculo optimizado w3

# variable con parte de los cálculos de forma auxiliar
# inventario inicial - min(inventario inicial, fcst w1) + inventario transito w2
left_part1 = (
    initial_inventory_w1
    - min_initial_inventory_w1_fcst_w1
    + inventory_transit_w2
)

# calculo de ordenes w1 (semana de análisis)
orders_w1 = fcst_w3 - left_part1 + np.minimum(left_part1, fcst_w2)

# puede ser que la orden sea negativa, porque sobra inventario, se pasa a cero
orders_w1 = orders_w1.clip(lower=0)

# generar output
data_submission.loc[:, "0"] = np.array(orders_w1)
data_submission = data_submission.reset_index()

# guardar
foldet_orders = "data/submission/orders"
week_index = 1
path_submission = f"{foldet_orders}/Week {week_index} - Submission.csv"
data_submission.to_csv(path_submission, index=False)

# revisar que se guardó
print("validar output guardado")
validar_output_guardado = pd.read_csv(path_submission)
validar_output_guardado
