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


""" 6. Tomar decisión ORDER DE LA SEMANA W1 """
# generar reglas - asume forecast perfectos y objetivo costo CERO en W+3"
data_submission = rules_systems_orders_perfect_forecast(
    previous_df_state=previous_data_state,
    df_fcst=data_fcst,
    df_submission=data_submission,
)

# guardar
foldet_orders = "data/submission/orders"
week_index = 1
path_submission = f"{foldet_orders}/Week {week_index} - Submission.csv"
data_submission.to_csv(path_submission, index=False)

# revisar que se guardó
print("validar output guardado")
validar_output_guardado = pd.read_csv(path_submission)
validar_output_guardado
