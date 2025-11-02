# %%
"""
Script con cálculo de métricas
"""

from sklearn.metrics import mean_absolute_error
import pandas as pd


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
