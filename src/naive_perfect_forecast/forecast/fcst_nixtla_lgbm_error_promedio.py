# %%
"""
Dado el backtest, que tiene multiples ejecuciones del modelo,
calcular el MAE de cada serie EN TEST (considerando h=1, h=2, h=3)

Al analizar las métricas en el backtest, en TRAIN (h=1) el error suele ser pequeño
para todas las series. PERO EN TEST/FUTURE (h=1, h=2, h=3) que es el usado para optimizar
los costos, el error el forecast es bastante mayor lo que provoca altos costos

Así se calcula el error en backtest TEST, para sumarlo como holgura, ya que el costo de venta perdida es 5 veces mayor al costo de sobrestock
"""

import pandas as pd

from utils.utils import set_root_path

from sklearn.metrics import mean_absolute_error


# set root repo
set_root_path()


""" 0. auxiliar function """


def calcular_mae_serie(df_metrics, features_columns, column_true, column_pred):
    """
    Calcular métrica MAE a nivel serie
    Interesa saber que series tienen más error, independiente si es un volumen alto o bajo
    (ya que cada producto que quede en stock o faltara stock es un costo asociado)
    """
    df_metrics = df_metrics.copy()

    # agrupar
    df_metrics_output = (
        df_metrics.groupby(features_columns)
        .apply(
            lambda g: pd.Series(
                {
                    "mae": mean_absolute_error(g[column_true], g[column_pred]),
                    "sum_y_true": g[column_true].sum(),
                    "mean_y_true": g[column_true].mean(),
                }
            )
        )
        .reset_index()
    )

    # ordenar de mayor a menor error (todas las series deben predecirse bien)
    df_metrics_output = df_metrics_output.sort_values("mae", ascending=False)

    return df_metrics_output


""" 2. read data forecast OUTPUT BACKTEST """
# obs: los datos de train y test ya tienen el merge con sus datos de true
folder_forecasts = "data/submission/backtest"

data_fcst_real_train_backtest = pd.read_parquet(
    f"{folder_forecasts}/data_fcst_real_train_backtest.parquet"
)
data_fcst_real_test_backtest = pd.read_parquet(
    f"{folder_forecasts}/data_fcst_real_test_backtest.parquet"
)


""" 3. Calcular MAE de cada serie - TEST de backtest - output de mayor a menor error """
mae_serie_test = calcular_mae_serie(
    df_metrics=data_fcst_real_test_backtest,
    features_columns=["unique_id"],
    column_true="y_true",
    column_pred="forecast_int",
)


""" 4. Guardar MAE de cada serie test de backtest - se utiliza como holgura v1 """

# holgura para pedir sobre-stock a lo predicho por el forecast, así evitar
# que el costo de shortage aumente, por el contrario aumentar el de holding
# cuyo costo es menor por unidad
folder_output = "data/submission/fcst"
mae_serie_test.to_parquet(f"{folder_output}/mae_serie_test.parquet")
