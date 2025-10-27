# %%
"""
General utils functions
"""

import os
import subprocess
import shutil
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


####### GENRAR DATA RAW week0, week1, week2, etc #######
# FUNCIONES PARA LEER DATA RAW WEEK0 Y DATA RAW INPUT SUBMISSIONS WEEK1,2,
# ETC Y GENERAR RAW EN FORMATO PARA SER CONSUMIDOS EN CUALQUIER SEMANA
def init_raw_to_models_week0():
    """
    Copiar y pegar files de input en 'data/input/raw_start_week0' y
    dejarlos en carpeta 'data/input/to_models' para ser procesados por modelos
    """

    set_root_path()

    # Rutas de origen y destino
    origen = "data/input/raw_start_week0"
    destino = "data/input/to_models"

    # Crear el destino si no existe
    os.makedirs(destino, exist_ok=True)

    # Recorrer los archivos del folder de origen
    for archivo in os.listdir(origen):
        ruta_origen = os.path.join(origen, archivo)
        ruta_destino = os.path.join(destino, archivo)

        # Solo copiar si es un archivo (no carpeta)
        if os.path.isfile(ruta_origen):
            shutil.copy2(ruta_origen, ruta_destino)

    return "data week 0 - creada en folder data/input/to_models "


def update_raw_to_models(week_index):
    """
    Actualizar datos raw to models.
    Dado el valor de 'week_index' tomar los archivos de la semana pasada
    (tanto raw como output de la submission previa) y actualizar raw
    de la semana siguiente para poder ser utilizados por el modelo
    """

    ##################################################
    # index previo
    prev_week_index = week_index - 1

    # valor fecha semana actual y previa
    date_week0 = "2024-04-08"

    date_prev_week = (
        pd.to_datetime(date_week0) + pd.Timedelta(days=7 * prev_week_index)
    ).strftime("%Y-%m-%d")

    date_current_week = (
        pd.to_datetime(date_week0) + pd.Timedelta(days=7 * week_index)
    ).strftime("%Y-%m-%d")

    ##################################################
    ########## copiar y pegar archivos semana previa que no sufren modificación
    set_root_path()

    # copiar y pegar data in stock - cambiar index week
    archivo_origen = (
        f"data/input/to_models/Week {prev_week_index} - In Stock.csv"
    )
    carpeta_destino = "data/input/to_models"
    nuevo_nombre = f"Week {week_index} - In Stock.csv"
    # os.makedirs(carpeta_destino, exist_ok=True)
    archivo_destino = os.path.join(carpeta_destino, nuevo_nombre)
    shutil.copy2(archivo_origen, archivo_destino)

    # copiar y pegar data master - cambiar index week
    archivo_origen = (
        f"data/input/to_models/Week {prev_week_index} - Master.csv"
    )
    carpeta_destino = "data/input/to_models"
    nuevo_nombre = f"Week {week_index} - Master.csv"
    # os.makedirs(carpeta_destino, exist_ok=True)
    archivo_destino = os.path.join(carpeta_destino, nuevo_nombre)
    shutil.copy2(archivo_origen, archivo_destino)

    # copiar y pegar data submission template - cambiar index week
    archivo_origen = f"data/input/to_models/Week {prev_week_index} - Submission Template.csv"
    carpeta_destino = "data/input/to_models"
    nuevo_nombre = f"Week {week_index} - Submission Template.csv"
    # os.makedirs(carpeta_destino, exist_ok=True)
    archivo_destino = os.path.join(carpeta_destino, nuevo_nombre)
    shutil.copy2(archivo_origen, archivo_destino)

    ##################################################
    ########## Actualizar STATE con el output submission semana previa

    # copiar y pegar state output del submission de semana previa
    archivo_origen = (
        f"data/input/raw_prev_submissions/output_salesw{week_index}.csv"
    )
    carpeta_destino = "data/input/to_models"
    nuevo_nombre = (
        f"Week {week_index} - {date_current_week} - Initial State.csv"
    )
    # os.makedirs(carpeta_destino, exist_ok=True)
    archivo_destino = os.path.join(carpeta_destino, nuevo_nombre)
    shutil.copy2(archivo_origen, archivo_destino)

    ##################################################
    ########## Actualizar Sales con la info de la nueva semana
    # cargar sales existente, agregar una nueva columna con la nueva venta
    # desde w1 en adelante (sales solo aparece en el nuevo state input)

    # copiar y pegar sales - cambiar index week
    archivo_origen = f"data/input/to_models/Week {prev_week_index} - {date_prev_week} - Sales.csv"
    carpeta_destino = "data/input/to_models"
    nuevo_nombre = f"Week {week_index} - {date_current_week} - Sales.csv"
    # os.makedirs(carpeta_destino, exist_ok=True)
    archivo_destino = os.path.join(carpeta_destino, nuevo_nombre)
    shutil.copy2(archivo_origen, archivo_destino)

    # leer archivo de sales y archivo state semana en curso
    folder = "data/input/to_models/"
    path_sales_current_week = (
        folder + f"Week {week_index} - {date_current_week} - Sales.csv"
    )
    path_state_current_week = (
        folder + f"Week {week_index} - {date_current_week} - Initial State.csv"
    )
    sales_current_week = pd.read_csv(path_sales_current_week)
    state_current_week = pd.read_csv(path_state_current_week)

    # actualizar SALES con la venta de la semana recién terminada
    # obtenida del state output de la competición
    sales_current_week.loc[:, date_current_week] = state_current_week["Sales"]

    # guardar nuevo archivo
    sales_current_week.to_csv(path_sales_current_week, index=False)

    # output
    return f"data RAW week {week_index} to models creada!! "


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
