"""
General utils functions
"""

import os
import subprocess
from pathlib import Path

import pandas as pd


def set_root_path():
    """
    Setear ubicación en el root del repo
    """

    repo_root = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "--show-toplevel"],  # noqa: S607 # noqa: S603
        capture_output=True,
        text=True,
    ).stdout.strip()

    ROOT = Path(repo_root)
    os.chdir(ROOT)
    print(f"root path: {ROOT}")


def read_input_data(week_index, date_index):
    """
    Leer archivos input RAW. Retornar en single index
    Params input: para diferenciar a qué semana corresponde los archivos
    """

    # params fijos
    INDEX = ["Store", "Product"]
    folder_data = "data/input"

    # ----------------------------
    # files con filtro "week_index" y "date_index"
    # read sales
    # path_sales = f"{folder_data}/Week 0 - 2024-04-08 - Sales.csv"
    path_sales = f"{folder_data}/Week {week_index} - {date_index} - Sales.csv"
    sales = pd.read_csv(path_sales).set_index(INDEX)
    sales.columns = pd.to_datetime(sales.columns)
    # sales = sales.reset_index()

    # read state
    # path_state = f"{folder_data}/Week 0 - 2024-04-08 - Initial State.csv"
    path_state = (
        f"{folder_data}/Week {week_index} - {date_index} - Initial State.csv"
    )
    state = pd.read_csv(path_state).set_index(INDEX)
    # state = state.reset_index()

    # ----------------------------
    # files solo con filtro "week"
    # read in_stock
    # path_in_stock = f"{folder_data}/Week 0 - In Stock.csv"
    path_in_stock = f"{folder_data}/Week {week_index} - In Stock.csv"
    in_stock = pd.read_csv(path_in_stock).set_index(INDEX)
    in_stock.columns = pd.to_datetime(in_stock.columns)
    # in_stock = in_stock.reset_index()

    # read master - info de los productos
    # path_master = f"{folder_data}/Week 0 - Master.csv"
    path_master = f"{folder_data}/Week {week_index} - Master.csv"
    master = pd.read_csv(path_master).set_index(INDEX)
    # master = master.reset_index()

    # read formato submission
    # path_submission = f"{folder_data}/Week 0 - Submission Template.csv"
    path_submission = (
        f"{folder_data}/Week {week_index} - Submission Template.csv"
    )
    submission = pd.read_csv(path_submission)

    print("data raw readed!")
    return sales, state, in_stock, master, submission


def read_preprocess_data(week_index=None, date_index=None):
    """
    Read data generada en step "0_preprocess"
    TODO: aún no tengo claro si se sobreescribe o es necesario el
    identificador de la semana
    """
    folder_data = "data/preprocess"

    # data (data_sales, data_in_stock) (para entrenar fcst)
    df = pd.read_parquet(f"{folder_data}/data.parquet")

    # data_state (para decidir cuánto pedir)
    df_state = pd.read_parquet(f"{folder_data}/data_state.parquet")

    # data_in_stock (contiene las proximas 8 fechas a forecast en competencia)
    # útil para evitar errores
    df_in_stock = pd.read_parquet(f"{folder_data}/data_in_stock.parquet")

    # data_master (para features exógenas para modelo - static)
    df_master = pd.read_parquet(f"{folder_data}/data_master.parquet")

    # data_submission (formato output)
    df_submission = pd.read_parquet(f"{folder_data}/data_submission.parquet")

    print("data preprocess readed!")
    return df, df_state, df_in_stock, df_master, df_submission


def read_processed_data(week_index=None, date_index=None):
    """
    Read data generada en step "1_fill_no_stock_data".
    Data processed - lista ser forecasteada y luego optimizar pedidos
    TODO: aún no tengo claro si se sobreescribe o es necesario el
    identificador de la semana
    """
    folder_data = "data/processed"

    # data (data_sales, data_in_stock) (para entrenar fcst)
    df = pd.read_parquet(f"{folder_data}/data.parquet")

    # data_state (para decidir cuánto pedir)
    df_state = pd.read_parquet(f"{folder_data}/data_state.parquet")

    # data_in_stock (contiene las proximas 8 fechas a forecast en competencia)
    # útil para evitar errores
    df_in_stock = pd.read_parquet(f"{folder_data}/data_in_stock.parquet")

    # data_master (para features exógenas para modelo - static)
    df_master = pd.read_parquet(f"{folder_data}/data_master.parquet")

    # data_submission (formato output)
    df_submission = pd.read_parquet(f"{folder_data}/data_submission.parquet")

    print("data processed readed!")
    return df, df_state, df_in_stock, df_master, df_submission
