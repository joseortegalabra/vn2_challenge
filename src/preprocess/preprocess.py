# %%

"""
PREPROCESS - LEER DATA RAW Y REALIZAR PROCESAMIENTO A LOS DATOS QUE ES TRANSVERSAL PARA TODOS LOS SIGUIENTES PASOS
"""

import numpy as np
import pandas as pd

from utils.utils import read_input_data, set_root_path

# set root repo
set_root_path()

""" 1. read raw data """
# leer archivos raw entregados por la competencia
data_sales, data_state, data_in_stock, data_master, data_submission = (
    read_input_data(week_index="0", date_index="2024-04-08")
)


""" 2. Reemplazar con NULL cuando no hubo venta por problemas de stock """
# These are shortages, we'll put missing data
# se reemplaza el dataframe existente
data_sales[~data_in_stock] = np.nan


""" 3. Melt de la data sales. Dejar en formato de serie de tiempo """
# melt
data_sales = data_sales.reset_index()
data = data_sales.melt(
    id_vars=["Store", "Product"], var_name="ds", value_name="y"
)

# generar unique_id
data["unique_id"] = (
    data["Store"].astype(str) + "-" + data["Product"].astype(str)
)

# correciones
data["ds"] = pd.to_datetime(data["ds"])

order_columns = ["unique_id", "ds", "y", "Store", "Product"]
data = data[order_columns]

data = data.sort_values(["unique_id", "ds"], ascending=[True, True])

# validar que no se pierde info
number_null_melt_data = data["y"].isnull().sum()
number_null_raw_data = data_sales.isnull().sum().sum()
assert (
    number_null_melt_data == number_null_raw_data
), f"Error: los valores nulos cambiaron ({number_null_melt_data} vs {number_null_raw_data})"

del data_sales


""" 4. Eliminar valores nulos - solo los nulos hasta que aparece la serie """
""" Eliminar nulos cuando aún no se vende el producto en la tienda """
data_copy = data.copy()


def drop_initial_nulls(group):
    """
    Para un grupo (groupby de pandas) encontrar la posición del primer valor
    NO NULO
    Retornar DataFrame desde el primer valor no NULL en adelante
    """
    # encontrar la posición (entero) del primer valor no nulo
    first_valid_pos = group["y"].first_valid_index()
    if first_valid_pos is None:
        # toda la serie es nula → devolver vacío
        return pd.DataFrame(columns=group.columns)
    else:
        # usar boolean mask para filtrar desde el primer no nulo
        mask = group.index >= first_valid_pos
        return group.loc[mask]


data = (
    data.groupby("unique_id", group_keys=False)
    .apply(drop_initial_nulls)
    .reset_index(drop=True)
)

# print informativo
print("eliminados los primeros NULL antes que se aparezca la primera venta!")
print("cantidad de nulos aún existentes", data["y"].isnull().sum())
print("cantidad de datos: ", data.shape)


# validar que no se pierden ventas
total_y_data_copy = data_copy["y"].sum()
total_y_data = data["y"].sum()
assert (
    total_y_data_copy == total_y_data
), f"Error: el total 'y' cambió ({total_y_data_copy} vs {total_y_data})"

# debugging - ver cómo cambia una serie al eliminar los nulos
# data_copy[data_copy["unique_id"] == "0-182"]
# data[data["unique_id"] == "0-182"]
del data_copy


""" 5. Eliminar los primeros ceros NO NULL """
# también aparecen valores cero al inicio de algunas series,
# PERO SÍ HABÍA STOCK
# por ahora se omite


""" 6. Procesamiento otros datos raw """
# data state - generar unique_id
data_state = data_state.reset_index()
data_state["unique_id"] = (
    data_state["Store"].astype(str) + "-" + data_state["Product"].astype(str)
)

cols = ["unique_id"] + [c for c in data_state.columns if c != "unique_id"]
data_state = data_state[cols]


# data master - generar unique_id
# esta data posteriormente puede ser utilizada como features exógenas
data_master = data_master.reset_index()
data_master["unique_id"] = (
    data_master["Store"].astype(str) + "-" + data_master["Product"].astype(str)
)

cols = ["unique_id"] + [c for c in data_master.columns if c != "unique_id"]
data_master = data_master[cols]


# data_in_stock - generar unique_id
# contiene las próximas 8 fechas que se deben forecastear para la competencia
# puede ser útil para asegurar que no haya errores
data_in_stock = data_in_stock.reset_index()
data_in_stock["unique_id"] = (
    data_in_stock["Store"].astype(str)
    + "-"
    + data_in_stock["Product"].astype(str)
)

cols = ["unique_id"] + [c for c in data_in_stock.columns if c != "unique_id"]
data_in_stock = data_in_stock[cols]


""" 7. Guardar datos """
folder_output = "data/preprocess"

# save data (data_sales) (para entrenar fcst)
data.to_parquet(f"{folder_output}/data.parquet")

# save data_state (para decidir cuánto pedir)
data_state.to_parquet(f"{folder_output}/data_state.parquet")

# save data_in_stock (contiene las próximas 8 fechas a forecastear)
data_in_stock.to_parquet(f"{folder_output}/data_in_stock.parquet")

# save data_master (para features exógenas para modelo - static)
data_master.to_parquet(f"{folder_output}/data_master.parquet")

# save data_submission (formato output)
data_submission.to_parquet(f"{folder_output}/data_submission.parquet")

print("data preprocess saved!")
# %%
