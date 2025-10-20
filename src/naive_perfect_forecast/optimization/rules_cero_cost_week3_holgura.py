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

from utils.models_optimization import format_forecast_to_optimization
from utils.models_optimization import rules_systems_orders_perfect_forecast

# set root repo
set_root_path()


""" 1. read "processed" data """
# leer archivos "processed" generados en step anterior
data, data_state, data_in_stock, data_master, data_submission = (
    read_processed_data(week_index="0", date_index="2024-04-08")
)


""" 2. read forecast generados """
folder_fcst = "data/submission/fcst"

data_fcst_real_train = pd.read_parquet(
    f"{folder_fcst}/data_fcst_real_train.parquet"
)
data_fcst_real_test = pd.read_parquet(
    f"{folder_fcst}/data_fcst_real_test.parquet"
)


""" 3. read MAE test - obtenido desde backtesting """
# leer mae, transformar a integer y utilizar como holgura
# columnas output "Store", "Product", "mae"
folder_output = "data/submission/fcst"
mae_serie_test = pd.read_parquet(f"{folder_output}/mae_serie_test.parquet")

mae_holgura_unique_id = mae_serie_test[["unique_id", "mae"]]

mae_holgura_unique_id[["Store", "Product"]] = (
    mae_holgura_unique_id["unique_id"].str.split("-", expand=True).astype(int)
)
mae_holgura_unique_id = mae_holgura_unique_id.drop(columns="unique_id")

mae_holgura_unique_id.loc[:, "mae"] = np.floor(mae_holgura_unique_id["mae"])
mae_holgura_unique_id["mae"] = mae_holgura_unique_id["mae"].astype(int)


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
data_fcst = format_forecast_to_optimization(
    df_fcst_real=data_fcst_real_test, df_state=previous_data_state
)


""" 6. Tomar decisión CUÁNTO ORDENAR en LA SEMANA W1 """
# generar reglas - asume forecast perfectos y objetivo costo CERO en W+3"
data_submission = rules_systems_orders_perfect_forecast(
    previous_df_state=previous_data_state,
    df_fcst=data_fcst,
    df_submission=data_submission,
)

# al output submission que se basa en fcst w+3.
# Agregar holgura basada en el MAE de TEST al realizar backtest de 1 año aprox
data_submission = pd.merge(
    data_submission,
    mae_holgura_unique_id,
    on=["Store", "Product"],
    how="left",
)
data_submission["0"] = data_submission["0"] + data_submission["mae"]
data_submission = data_submission.drop(columns="mae")

# guardar
foldet_orders = "data/submission/orders"
week_index = 1
path_submission = f"{foldet_orders}/Week {week_index} - Submission.csv"
data_submission.to_csv(path_submission, index=False)

# revisar que se guardó
print("validar output guardado")
validar_output_guardado = pd.read_csv(path_submission)
validar_output_guardado
