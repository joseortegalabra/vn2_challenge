# %%
"""
COMPLETAR DATOS DE VENTAS NULL POR NO STOCK
Hay datos de venta que faltan por no haber stock.
- Si en el real sobró stock, se sabe que esa es la demanda real para ese producto en esa fecha
- PERO si no había stock, no se vendió nada y por lo tanto hay una demanda potencial perdida, en caso de que hubiera stock se hubiera vendido algo.

VERSIÓN SIMPLE, SE RELLENAR LOS NULOS CON INTERPOLACIONES (interpolate, bfill, ffill, etc)
"""

# import numpy as np
# import pandas as pd

from utils.utils import read_preprocess_data, set_root_path


# TODO: aquí voy haciendo interpolación para llegar a una solución más rápida


# set root repo
set_root_path()

""" 1. read raw data """
# leer archivos raw entregados por la competencia
data, data_state, data_master, data_submission = read_preprocess_data(
    week_index="0", date_index="2024-04-08"
)
