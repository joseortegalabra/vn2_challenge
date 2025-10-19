# %%
"""
Realizar backtest del modelo. Backtest enfocado para evaluar estrategia de
control de inventarios.
- Entrenar modelo para cada fecha 1 hasta N. Guardar forecast realizado
hasta horizonte H.
- Con eso se puede simular resultados de la estrategia
- Al ser pocos datos, es factible entrenar un modelo para cada fecha, ya que
el tiempo de entrenamiento es muy corto
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
data, data_state, data_in_stock, data_master, data_submission = (
    read_processed_data(week_index="0", date_index="2024-04-08")
)


""" 2. Definir params para forecast """
horizonte_fcst = 3

# True: cuando se está haciendo pruebas/backtest/etc
# False: productivo entrenando con toda la historia disponible. Output challenge
develop = True


""" 3. Agregar columnas data_master. Static features """
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


""" 4. Generar features exógenas relacionadas a fechas """
# indicar de qué semana es con respecto al mes (primera semana del mes, segunda, etc)
data["week_of_month"] = data["ds"].apply(lambda d: (d.day - 1) // 7 + 1)


# ------ desde aquí cambia vs el código de TRAIN
# porque se necesita entrenar y guardar fcst de N ejecuciones ------ #


""" 5. Definir intervalo de fechas para generar data TRAIN para backtest """
# listado con todas las fechas
list_all_ds = data["ds"].unique()

# fecha fin (se dejan fechas fuera de train para poder evaluar horizonte fcst)
# las h fechas definidas en horizonte_fcst, +1 ya que train corta una semana antes
fecha_fin_train_backtest = list_all_ds[-(horizonte_fcst + 1)]

# fecha inicio
# la primera fecha que se puede forecastear considerando todos los lags para que haya mínimo 1 instancia usada para entrenamiento)
# max_lags = 8
# max_ma_lags = 12
# number_total_lags = max_lags + max_ma_lags
# fecha_inicio_train_backtest = list_all_ds[number_total_lags]

# opcion 2: tomar como fecha, la primera que permita generar HISTORIA DE TODAS LAS SERIES
# las series más nuevas, necesitan un mínimo de x historia para poder entrenarse por los lags
# nixlta igual realiza INFERENCIA al ser modelo global, pero al no haber historia de la serie, los fcst pueden ser peor
# ADEMÁS PERMITE TENER MÁS HISTORIA DE LAS SERIES LARGAS
# str_fecha_inicio_train_backtest = "2022-12-26 00:00:00"
str_fecha_inicio_train_backtest = "2023-06-05 00:00:00"
fecha_inicio_train_backtest = pd.to_datetime(str_fecha_inicio_train_backtest)

# fecha inicio de todos los datos
fecha_incio_all = data["ds"].min()

# generar lista recortada de fechas. Lista con las fechas que se realiza inferencia
# fechas en la que se entrena cada modelo
list_all_ds_recortada = [
    d
    for d in list_all_ds
    if fecha_inicio_train_backtest <= d <= fecha_fin_train_backtest
]


""" 6. Generar N conjuntos de datos train y test - realizar backtest """
# INICIALIZAR Dataframe output
# KEY ACTUALIZACIÓN SEMANA IDENTIFICADA COMO W0 (última semana de TRAIN)
# en forecast se guarda el forecast de w1, w2, w3
data_fcst_real_train_backtest = pd.DataFrame()
data_fcst_real_test_backtest = pd.DataFrame()


""" 6.1 Crear un for para recorrer los N horizontes de train y test """

for index in range(len(list_all_ds_recortada)):
    """ 6.2 filtrar lista con diferentes fechas a realizar forecast (week 0) """
    date_week0 = list_all_ds_recortada[index]
    print(f"fecha w0 - fecha fin TRAIN: {index} // {date_week0}")

    """ 6.5. Definir qué es Train y Test. Genear columna que lo identidique"""
    # generar fechas inicio, fin de train y test de acuerdo al "FILTRO" de week 0
    # train: siempre desde inicio all data hasta fecha de corte w0
    # test: w1, w2, w3
    fecha_incio_filter_train = fecha_incio_all
    fecha_fin_filter_train = date_week0
    fecha_incio_filter_test = date_week0 + pd.Timedelta(days=7)
    fecha_fin_filter_test = date_week0 + pd.Timedelta(days=7 * horizonte_fcst)

    # crear columna para marcar TRAIN y TEST.
    # generar dataframe filtrado realizar forecast de w1, w2, w3 de backtest
    data_filter = data.copy()

    mask_filter_train = (data_filter["ds"] >= fecha_incio_filter_train) & (
        data_filter["ds"] <= fecha_fin_filter_train
    )
    data_filter.loc[mask_filter_train, "TRAIN_FUTURE"] = "TRAIN"

    mask_filter_test = (data_filter["ds"] >= fecha_incio_filter_test) & (
        data_filter["ds"] <= fecha_fin_filter_test
    )
    data_filter.loc[mask_filter_test, "TRAIN_FUTURE"] = "FUTURE"

    """ 6.6 Separar en data train y data test. en base a columna "TRAIN_FUTURE" """
    data_filter_train, data_filter_test, data_filter_test_exog = (
        split_train_test_using_column_mark(df=data_filter, verbose=True)
    )

    """ 6.7 Generar modelo """
    # generar modelo, entrenar y predecir TRAIN y TEST/FUTURE
    data_filter_fcst_real_train, data_filter_fcst_real_test = (
        train_predict_ts_mlforecast(
            df_train=data_filter_train,
            df_test=data_filter_test,
            df_test_exog=data_filter_test_exog,
            horizonte_fcst=horizonte_fcst,
        )
    )

    # # debugging - asegurar que en TRAIN se generan features de todas las series
    # # que se van a predecir en test
    # # ver qué pasa cuando son menos series
    # series_train = data_filter_fcst_real_train["unique_id"].unique().tolist()
    # series_test = data_filter_fcst_real_test["unique_id"].unique().tolist()
    # print(
    #     "series que están en test, pero no se generó historia al ser series recientes",
    #     len(set(series_test) - set(series_train)),
    # )

    """ 6.8 realizar append en dataframe output """
    # agregan index - key actualización semana WO
    data_filter_fcst_real_train["week0_update"] = date_week0
    data_filter_fcst_real_test["week0_update"] = date_week0

    # realizar append
    data_fcst_real_train_backtest = pd.concat(
        [data_fcst_real_train_backtest, data_filter_fcst_real_train],
        ignore_index=False,
    )

    data_fcst_real_test_backtest = pd.concat(
        [data_fcst_real_test_backtest, data_filter_fcst_real_test],
        ignore_index=False,
    )


""" 7. Guardar output """
# guardar forecast (real y fcst) generados (train y test)
folder_output = "data/submission/backtest"

data_fcst_real_train_backtest.to_parquet(
    f"{folder_output}/data_fcst_real_train_backtest.parquet"
)
data_fcst_real_test_backtest.to_parquet(
    f"{folder_output}/data_fcst_real_test_backtest.parquet"
)


""" 8. Calcular métricas """
# calcular MAE (con forecast decimal y con forecast int)
print("METRICS GLOBAL")

# train - horizonte de fcst h=1
mae_train = mean_absolute_error(
    y_true=data_fcst_real_train_backtest["y_true"],
    y_pred=data_fcst_real_train_backtest["forecast"],
)
mae_train_int = mean_absolute_error(
    y_true=data_fcst_real_train_backtest["y_true"],
    y_pred=data_fcst_real_train_backtest["forecast_int"],
)
print("mae_train: ", mae_train)
print("mae_train_int: ", mae_train_int)


# test - horizonte de fcst h=3
mae_test = mean_absolute_error(
    y_true=data_fcst_real_test_backtest["y_true"],
    y_pred=data_fcst_real_test_backtest["forecast"],
)
mae_test_int = mean_absolute_error(
    y_true=data_fcst_real_test_backtest["y_true"],
    y_pred=data_fcst_real_test_backtest["forecast_int"],
)
print("mae_test: ", mae_test)
print("mae_test_int: ", mae_test_int)
# %%
