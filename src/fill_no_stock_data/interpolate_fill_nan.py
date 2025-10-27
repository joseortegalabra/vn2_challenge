# %%
"""
COMPLETAR DATOS DE VENTAS NULL POR NO STOCK
Hay datos de venta que faltan por no haber stock.
- Si en el real sobró stock, se sabe que esa es la demanda real para ese producto en esa fecha
- PERO si no había stock, no se vendió nada y por lo tanto hay una demanda potencial perdida, en caso de que hubiera stock se hubiera vendido algo.

VERSIÓN SIMPLE, SE RELLENAR LOS NULOS CON INTERPOLACIONES (interpolate, bfill, ffill, etc)
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import read_preprocess_data, set_root_path

# set root repo
set_root_path()


""" 1. read "preprocess" data """
# leer archivos "preprocess" generados en step anterior
param_index_current_week = "1"
param_value_current_date = "2024-04-15"
data, data_state, data_in_stock, data_master, data_submission = (
    read_preprocess_data(
        week_index=param_index_current_week,
        date_index=param_value_current_date,
    )
)


""" 2. Completar valores null usando interpolación """
data_copy = data.copy()

# print cantidad de nulos iniciales
print("info data original")
print("cantidad de nulos: ", data_copy["y"].isnull().sum())
print("shape data: ", data_copy.shape)
info_percent_null = np.round(
    100 * data_copy["y"].isnull().sum() / data_copy.shape[0], 2
)
print(f"porcentaje de nulos: {info_percent_null} %")
print("volumen total de ventas: ", data_copy["y"].sum())


# interpolar. con valores enteros. que aproxime al entero más cercano
data = data.sort_values(["unique_id", "ds"])

data["y"] = (
    data.groupby("unique_id")["y"]
    .transform(lambda x: x.interpolate(limit_direction="both"))
    .round()
    # .astype(int)
)


# validar - que no se cree ni se pierda ninguna fila ni columna
shape_data_copy = data_copy.shape
shape_data = data.shape
assert shape_data_copy == shape_data, (
    f"Error: shape del DataFrame cambió. "
    f"Antes: {shape_data_copy}, ahora: {shape_data}"
)

# print info
print("info luego fill null")
print("cantidad de nulos: ", data["y"].isnull().sum())
print("shape data: ", data.shape)
info_percent_null = np.round(100 * data["y"].isnull().sum() / data.shape[0], 2)
print(f"porcentaje de nulos: {info_percent_null} %")
print("volumen total de ventas: ", data["y"].sum())


"""
TODO: falta validar si todas las series tienen todos sus valores
desde su fecha de inicio hasta la última fecha registrada
"""


""" 3. Save data processed - lista para ser usada en modelamiento """
# ojo se guardan también data_state, data_master aunque no hayan tenido cambios
folder_output = "data/processed"

data.to_parquet(f"{folder_output}/data.parquet")
data_state.to_parquet(f"{folder_output}/data_state.parquet")
data_in_stock.to_parquet(f"{folder_output}/data_in_stock.parquet")
data_master.to_parquet(f"{folder_output}/data_master.parquet")
data_submission.to_parquet(f"{folder_output}/data_submission.parquet")
print("data processed saved!")


""" 4. Extra """

###### ---- TODO: extra solo informativo, quizás en el futuro se necesite
###### comparar otros métodos de fill null y ver resultados visualmente
# graficar una serie de ejemplo
# graficar tendencia luego de fill null.
# además graficar punto rojo cuando los valores originalmente fueron NULL

# id to plot
unique_id_plot = "29-17"

# filtar serie
data_copy_plot = data_copy[data_copy["unique_id"] == unique_id_plot]
data_plot = data[data["unique_id"] == unique_id_plot]

# graficar
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_plot, x="ds", y="y", marker="o", label="value y")

missing_points = data_copy_plot[data_copy_plot["y"].isna()]
for ds_missing in missing_points["ds"]:
    plt.axvline(
        x=ds_missing,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="_nolegend_",
    )

plt.title("Trend Sales")
plt.legend()
plt.show()
