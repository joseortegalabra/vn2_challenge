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
horizonte_fcst = 8  # predicho las próximas X semanas

# si es True, es porque se está desarrollando
# y se dejan datos test para medir métricas
develop = True


'''

""" 3. Definir corte train y test. Si es desarrollo, el test es REAL """
if develop:
    # definir fecha corte 90% train y 10% test
    unique_dates = np.sort(data["ds"].unique())
    cut_idx = int(len(unique_dates) * 0.9)
    cut_date = unique_dates[cut_idx]

    # cortar datos para train y test
    data_train = data[data["ds"] <= cut_date]
    data_test = data[data["ds"] > cut_date]
else:
    # si es productivo - output competencia - usar todos los datos para train
    data_train = data
    data_test = pd.DataFrame()

# print info
print("train shape: ", data_train.shape)
print("test shape: ", data_test.shape)
print("train min date: ", data_train["ds"].min())
print("train max date: ", data_train["ds"].max())
'''

""" 4. """

# prod
# obtener la última fecha y sumar N fechas de acuerdo al horizonte fcst
# test corresponde a las predicciones futuras que se desconoce el real
last_date = data["ds"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=7),
    periods=horizonte_fcst,
    freq="W",
)

# generar dataframe test - por cada "unique_id"
