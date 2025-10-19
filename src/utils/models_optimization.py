"""
Script con los códigos para optimizar inventario a partir de un forecast
Pueden ser sistemas de reglas, modelos, etc
"""

import pandas as pd
import numpy as np


def update_state_true_demand(
    previous_df_state, df_fcst_real, df_order, df_fcst
):
    """
    Dado el estado previo (w0), la decisón de cuánto ordenar y la venta real,
    ACTUALIZAR STATE CON EL QUE SE CIERRA LA SEMANA W1
    (actualizando ventas, ventas no realizadas, productos en tránsito, etc)

    Args:
    df_fcst_real (DataFrame): df con el real y fcst. TEST. hasta horizonte h. INTERESA PARA EL REAL
    previous_df_state (DataFrame): df con el state previo (semana w0)
    df_order (DataFrame): df con el submission (la orden realizada en la semana w1)
    df_fcst (DataFrame): df con los valores forecasteados (SOLO SE AGREGA POR INFORMATIVO)

    Return:
    next_df_state (DataFrame): df con el state resultante (semana w1), usando las ventas reales
    """
    previous_df_state = previous_df_state.copy()
    df_fcst_real = df_fcst_real.copy()
    df_order = df_order.copy()
    df_fcst = df_fcst.copy()

    # obtener real con el que cerró W1.
    # CURVA DE DEMADA
    # así si el fcst da MAL, la estrategia va a dar mal (sobre stock o substock)
    date_w1 = df_fcst_real["ds"].min()
    demand = df_fcst_real[df_fcst_real["ds"] == date_w1]
    demand = demand[["unique_id", "ds", "y_true"]]

    # ordenar la DEMANDA en el mismo orden que "state"
    orden_unique_ids = previous_df_state["unique_id"].unique()
    demand["unique_id"] = pd.Categorical(
        demand["unique_id"], categories=orden_unique_ids, ordered=True
    )
    demand = demand.sort_values("unique_id").reset_index(drop=True)

    ####### CREAR NEXT_DF_STATE #######
    # generar dataframe next state y comenzar a completar columnas
    next_df_state = previous_df_state[["unique_id", "Store", "Product"]].copy()

    # inventario inicial
    next_df_state["Start Inventory"] = (
        previous_df_state["End Inventory"]
        + previous_df_state["In Transit W+1"]
    )

    # ventas (mínimo entre la demanda y el stock disponible)
    next_df_state["Sales"] = next_df_state["Start Inventory"].clip(
        upper=demand["y_true"]
    )

    # ventas perdidas por no tener stock
    next_df_state["Missed Sales"] = demand["y_true"] - next_df_state["Sales"]

    # inventario final: inventario inicial - ventas
    next_df_state["End Inventory"] = (
        next_df_state["Start Inventory"] - next_df_state["Sales"]
    )

    # calcular en tránsito w+1 (lo que venía en tránsito w2 del estado previo)
    next_df_state["In Transit W+1"] = previous_df_state["In Transit W+2"]

    # calcular en tránsito w+2 (LA ORDEN DECIDIDA POR EL MODELO)
    next_df_state["In Transit W+2"] = df_order["0"]

    # calcular costos y costos acumulados
    HOLDING_COST = 0.2
    SHORTAGE_COST = 1
    next_df_state["Holding Cost"] = (
        next_df_state["End Inventory"] * HOLDING_COST
    )
    next_df_state["Shortage Cost"] = (
        next_df_state["Missed Sales"] * SHORTAGE_COST
    )
    next_df_state["Cumulative Holding Cost"] = (
        previous_df_state["Cumulative Holding Cost"]
        + next_df_state["Holding Cost"]
    )
    next_df_state["Cumulative Shortage Cost"] = (
        previous_df_state["Cumulative Shortage Cost"]
        + next_df_state["Shortage Cost"]
    )

    # agregar columna adicional - SOLO INFORMATIVA - LOS VALORES FORECASTEADOS EN BACKTEST
    next_df_state = pd.merge(
        next_df_state, df_fcst, on=["unique_id"], how="left"
    )

    # agregar info adicional - valor real de demanda

    # print info costos
    info_holding_cost = next_df_state["Holding Cost"].sum().sum()
    info_shortage_cost = next_df_state["Shortage Cost"].sum().sum()
    info_round_cost = (
        next_df_state[["Holding Cost", "Shortage Cost"]].sum().sum()
    )
    info_cumulative_cost = (
        next_df_state[["Cumulative Holding Cost", "Cumulative Shortage Cost"]]
        .sum()
        .sum()
    )
    print(
        f"end week: {date_w1} // holding_cost: {info_holding_cost} // shortage_cost: {info_shortage_cost}  // round_cost: {info_round_cost} // cumulative_cost: {info_cumulative_cost}"
    )

    # retornar el df state resultante de W1. cierre de la semana con la venta REAL
    return next_df_state


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

    return df_submission
