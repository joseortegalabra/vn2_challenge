"""
Script auxiliar con códigos relacionadas a Modelos de Forecasting
"""

import numpy as np
import pandas as pd
from mlforecast import MLForecast
import lightgbm as lgb

from mlforecast.lag_transforms import (
    RollingMean,
    SeasonalRollingMean,
)


def split_train_test_using_column_mark(df, verbose=False):
    """
    Dado un datraframe "df" con la columna "TRAIN_FUTURE", separar en 2
    dataframes de acuerdo a la marca. Obtener datos train, test, test_exógenas

    Output
        df_train (DataFrame): dataframe con los datos de train
        df_test (DataFrame): dataframe con los datos de test
        df_test_exog (DataFrame): dataframe TEST, eliminando "y". Conservando las features exógenas para hacer forecast
    """
    df = df.copy()

    # separar por columna auxiliar que marca qué es train y qué es future
    mask_train = df["TRAIN_FUTURE"] == "TRAIN"
    df_train = df[mask_train]
    mask_test = df["TRAIN_FUTURE"] == "FUTURE"
    df_test = df[mask_test]

    # eliminar columnas auxiliares que no son features ni target
    list_columns_to_drop = ["TRAIN_FUTURE"]
    df_train = df_train.drop(columns=list_columns_to_drop)
    df_test = df_test.drop(columns=list_columns_to_drop)

    # Print info fechas y shape
    if verbose:
        print("DATA shape: ", df.shape)
        print("DATA TRAIN shape: ", df_train.shape)
        print("DATA TEST shape: ", df_test.shape)

        print("TRAIN min date: ", df_train["ds"].min())
        print("TRAIN max date: ", df_train["ds"].max())

        print("TEST min date: ", df_test["ds"].min())
        print("TEST max date: ", df_test["ds"].max())

    # Generar dataframe test - exógenas features - las necesarias para predecir
    # eliminar columna "y" en data test. Lo que se busca predecir
    df_test_exog = df_test.copy()
    df_test_exog = df_test_exog.drop(columns="y")

    return df_train, df_test, df_test_exog


def train_predict_ts_mlforecast(
    df_train, df_test, df_test_exog, horizonte_fcst
):
    """
    Entrenar y generar forecast con modelo de LGBM MLFORECAST
    Obs: internamenente se define el modelo, hiperparámetros, train e inferencia
    Obs 2: se ajusta el forecast para ser entero y no negativo
    Obs 3: se entrega dataframe con forecast y real. Si es TRAIN, el es forecast
    con horizonte h=1, mientras que si es TEST/FUTURE es el forecast con horizonte h=h
    ya que solo el test/future/forecast se hace la inferencia por recursividad

    # ejemplo M4 usando datos horarios: https://www.kaggle.com/code/lemuz90/m4-competition ejemplo M5 usando datos diario: https://www.kaggle.com/code/lemuz90/m5-mlforecast-eval
    """

    # params
    df_train = df_train.copy()
    df_test_exog = df_test_exog.copy()
    df_test = df_test.copy()

    # hiperparametros modelo LGBM - baseline y funciona bien
    params_lgbm = {
        "verbose": -1,
        "num_threads": 4,
        "force_col_wise": True,
        "num_leaves": 30,
        "n_estimators": 350,
        "random_state": 42,
    }

    # "hiperparámetros" features lags creados por nixtla
    # se prueba con un rezago de las ventas de los últimos 2 meses
    max_lags = 8
    list_lags = list(range(1, max_lags + 1))

    # crear modelo nixtla
    model_fcst = MLForecast(
        models=[lgb.LGBMRegressor(**params_lgbm)],
        freq="W-MON",  # semana comienza el lunes
        lags=list_lags,
        date_features=["month", "quarter", "week", "year"],
        lag_transforms={
            1: [
                RollingMean(4),  # media movil 1 mes
                RollingMean(8),  # media movil 2 meses
                RollingMean(12),  # media movil 3 meses
                SeasonalRollingMean(4, 4),  # estacionalidad mensual
                # ExpandingMean(),  # depende de toda la historia, mejor no usar
            ],
            4: [
                RollingMean(4),
                RollingMean(8),
                RollingMean(12),
                SeasonalRollingMean(4, 4),  # estacionalidad mensual
            ],
            8: [
                RollingMean(4),
                RollingMean(8),
                RollingMean(12),
                SeasonalRollingMean(4, 4),  # estacionalidad mensual
            ],
        },
        num_threads=-1,
    )

    # train
    model_fcst.fit(
        df_train,
        static_features=[],
        # se habilita la opción para obtener los valores forecasteado para los datos de train
        fitted=True,
    )

    # obs: obtener las features input para entrwnar el modelo
    # model_fcst.preprocess(df_train, static_features=[])

    # obs: mostrar las features usadas por el modelo
    # model_fcst.ts.features_order_

    # debugging - entender qué combinación faltó en data test exógena
    # debugging = model_fcst.get_missing_future(
    #    h=horizonte_fcst, X_df=df_test_exog
    # )

    # predecir
    predictions = model_fcst.predict(
        h=horizonte_fcst,
        X_df=df_test_exog,  # exógenas necesarias para predecir
        # new_df=data_train_future, # necesario cuando se entrena el modelo y se guarda y se hace inferencia con otros datos
    )

    # predecir datos fitten con data TRAIN (forecast horizonte h=1)
    fitted_values_train = model_fcst.forecast_fitted_values()

    # generar dataframe formato output para TRAIN y TEST
    # (incluyendo real y fcst. para train h=1, para test/future h=h)
    # train
    df_fcst_true_output_train = fitted_values_train.copy()
    df_fcst_true_output_train = df_fcst_true_output_train.rename(
        columns={"y": "y_true"}
    )
    df_fcst_true_output_train = df_fcst_true_output_train.rename(
        columns={"LGBMRegressor": "forecast"}
    )

    # future # TODO: quizás después separar para que entrege SOLO FCST
    # incluir df con True value "y"
    df_fcst_true_output_test = df_test.copy()
    df_fcst_true_output_test = df_fcst_true_output_test[
        ["unique_id", "ds", "y"]
    ]
    df_fcst_true_output_test = df_fcst_true_output_test.rename(
        columns={"y": "y_true"}
    )
    df_fcst_true_output_test = pd.merge(
        df_fcst_true_output_test,
        predictions,
        on=["unique_id", "ds"],
        how="left",
    )
    df_fcst_true_output_test = df_fcst_true_output_test.rename(
        columns={"LGBMRegressor": "forecast"}
    )

    # transformar fcst:
    # - si es menor a cero, llevar a cero
    # - redondear a int (ya sea redondeando o llevando al entero superior)

    df_fcst_true_output_train.loc[
        df_fcst_true_output_train["forecast"] <= 0, "forecast"
    ] = 0
    # df_fcst_true_output_train["forecast_int"] = np.ceil(
    df_fcst_true_output_train["forecast_int"] = np.round(
        df_fcst_true_output_train["forecast"]
    )

    df_fcst_true_output_test.loc[
        df_fcst_true_output_test["forecast"] <= 0, "forecast"
    ] = 0
    # df_fcst_true_output_test["forecast_int"] = np.ceil(
    df_fcst_true_output_test["forecast_int"] = np.round(
        df_fcst_true_output_test["forecast"]
    )

    return df_fcst_true_output_train, df_fcst_true_output_test
