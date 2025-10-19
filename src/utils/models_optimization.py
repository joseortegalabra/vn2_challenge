"""
Script con los códigos para optimizar inventario a partir de un forecast
Pueden ser sistemas de reglas, modelos, etc
"""

import pandas as pd
import numpy as np


def format_forecast_to_optimization(df_fcst_real, df_state):
    """
    Dado un df con forecast con el formato "unique_id", "ds", "y"
    Transformar a:
    - Formato "unique_id", "fcst_w1", "fcst_w2", "fcst_w3" etc (pivotear)
    - Ordenar "unique_id" para que tenga mismo orden que tabla "state" y "output"

    Args:
        df_fcst_real (DataFrame): df con el real y el fcst. TEST. INFERENCIA
        df_state (DataFrame): df con el formato del state

    Output:
        df_fcst(DataFrame): DataFrame pivoteado "unique_id", "fcst_w1", "fcst_w2", etc
    """
    df_fcst_real = df_fcst_real.copy()

    # pivotear: filas: unique_id, columnas: ds, values: forecast
    values_y_fcst = "forecast_int"
    df_fcst = df_fcst_real.pivot(
        index="unique_id", columns="ds", values=values_y_fcst
    )

    # renombrar columnas fcst_w1, fcst_w2, fcst_w3
    df_fcst.columns = ["fcst_w1", "fcst_w2", "fcst_w3"]
    df_fcst = df_fcst.reset_index()

    # Ordenar fcst en el mismo orden que previus_data_state
    orden_unique_ids = df_state["unique_id"].unique()

    df_fcst["unique_id"] = pd.Categorical(
        df_fcst["unique_id"], categories=orden_unique_ids, ordered=True
    )

    df_fcst = df_fcst.sort_values("unique_id").reset_index(drop=True)

    # AUX. Asegurar que state y fcst tienen el mismo orden de keys
    assert (
        df_fcst["unique_id"].values == df_state["unique_id"].values
    ).all(), "¡Los unique_id no están alineados entre df_fcst y df_state!"

    return df_fcst


def rules_systems_orders_perfect_forecast(
    previous_df_state, df_fcst, df_submission
):
    """
    Dado el previo state (conocer el inventario final y el flujo en tránsito w1, w2)
    y forecast, generar reglas a pedir en w1 (que llega para w3)
    OBS: se predice la futura demanda y se ordena en base a lo que se espera vender,
    así para que funcione se necesita un forecast perfecto
    """
    previous_df_state = previous_df_state.copy()
    df_fcst = df_fcst.copy()
    df_submission = df_submission.copy()

    """ 1. crear variables con los nombres de acuerdo a la fórmula calculada """
    # inventario inicial week 1
    initial_inventory_w1 = (
        previous_df_state["End Inventory"]
        + previous_df_state["In Transit W+1"]
    )

    # inventario en tránsito week 2 (el de week 1 se suma como inventario inicial w1)
    inventory_transit_w2 = previous_df_state["In Transit W+2"]

    # forecast predichos
    fcst_w1 = df_fcst["fcst_w1"]
    fcst_w2 = df_fcst["fcst_w2"]
    fcst_w3 = df_fcst["fcst_w3"]

    """ 2. generar reglas cálculo optimizado w3 - sin stock """
    # mínimo entre inventario incial w1 y fcst demanda w1 (se vende lo se acabe primero)
    min_initial_inventory_w1_fcst_w1 = -np.minimum(
        initial_inventory_w1, fcst_w1
    )

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
    df_submission.loc[:, "0"] = np.array(orders_w1)
    df_submission = df_submission.reset_index()

    return df_submission
