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

from utils.models_optimization import format_forecast_to_optimization
from utils.models_optimization import rules_systems_orders_perfect_forecast
from utils.models_optimization import update_state_true_demand


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


""" 5. Realizar FOR por cada ejecución del foreast y calcular ORDEN y
calcular resultados YA QUE SE TIENE EL REAL
- realizar orden en base al fcst y evaluar vs el real """
# output donde se guarda todos los "state" resultante del backtest
next_data_state_backtest = pd.DataFrame()


# obtener listado con todas las ejecuciones del backtest
list_ds_ejecuciones_fcst = (
    data_fcst_real_test_backtest["week0_update"].unique().tolist()
)

# for de cada una de las ejecuciones
for date_week0 in list_ds_ejecuciones_fcst:
    print(f"Semana w0: {date_week0}")

    # filtrar forecast de la fecha de ejecución filtrada
    data_fcst_real_test_backtest_filtered = data_fcst_real_test_backtest[
        data_fcst_real_test_backtest["week0_update"] == date_week0
    ]

    """ 5. Dar formato forecast generado - fechas en las columnas """
    # OJO: la data FILTERED de la ejecución DESEADA
    data_fcst = format_forecast_to_optimization(
        df_fcst_real=data_fcst_real_test_backtest_filtered,
        df_state=previous_data_state,
    )

    """ 6. Tomar decisión CUÁNTO ORDENAR en LA SEMANA W1 """
    # generar reglas - asume forecast perfectos y objetivo costo CERO en W+3"
    data_submission = rules_systems_orders_perfect_forecast(
        previous_df_state=previous_data_state,
        df_fcst=data_fcst,
        df_submission=data_submission,
    )

    """ 7. actualizar STATE CIERRE W1. USANDO LOS REALES DE VENTA """
    next_data_state = update_state_true_demand(
        df_fcst_real=data_fcst_real_test_backtest_filtered,
        previous_df_state=previous_data_state,
        df_order=data_submission,
        df_fcst=data_fcst,
    )

    """ 8. Reemplazar previo state con el state nuevo """
    # reemplazar state PREVIO (w0) con state fin de la semana (w1)
    # continuar de forma iterativa
    previous_data_state = next_data_state

    """ 8. Append "next_state" en df que guarda "STATE" de cada ejecución del backtest - debugging - info """
    next_data_state_aux = next_data_state.copy()
    next_data_state_aux["week0_update"] = date_week0

    next_data_state_backtest = pd.concat(
        [next_data_state_backtest, next_data_state_aux],
        ignore_index=False,
    )


""" 9. Guardar output backtest estrategia """
folder_output = "data/submission/backtest"
next_data_state_backtest.to_csv(
    f"{folder_output}/next_data_state_backtest.csv"
)

# debugging - revisar un sku en especifico - ej el de mayor volumen
unique_id_filter = "61-124"
debugging_backtest = next_data_state_backtest[
    next_data_state_backtest["unique_id"] == unique_id_filter
]
debugging_backtest.to_csv(f"{folder_output}/debugging_backtest.csv")
