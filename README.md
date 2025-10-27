# vn2_challenge
Repo with code for VN2 competition

Orden de ejecución carpeta SRC
Orden 0: preproces
    - update_raw.py (cambiar manualmente el índice de la semana en curso)
    - preprocess.py

Orden 1: fill_no_stock_data

Orden 2: naive_perfect_forecast
    - forecast
        - backtest: realizar backtest del modelo
        - error promedio: a partir del backtest, calcular el error promedio (MAE) de cada serie (unique_id). Este error es utilizado para la política de control de inventario basado en forecast
        - fcst_nixtla_lgbm: marcar como prod. Realizar el entrenamiento del modelo con TODA la historia existente y generar forecast para horizonte h=3
    - optimization
