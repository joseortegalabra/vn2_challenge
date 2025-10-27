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
        - Existe 2 algoritmos de optimización basados en que el costo a la semana w+3 sea cero. El primero originado de la idea de tener un forecast 100% preciso. Como esto no es factible, existe un segundo algoritmo basado en la idea del forecast 100% preciso, pero que se suma una holgura (ya que el costo de sobrestimar es menor al costo de subestimar), donde esta holgura corresponde al MAE obtenido en el backtest del forecast
        - Existen 2 script, backtest para ver el error que obtendría cada estrategia y un script output que genera la orden de inventario para la semana en curso (SUBMISSION CHALLENGE)
