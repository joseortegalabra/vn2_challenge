# %%
"""
Naive Optimization
Basada en forecast y reglas
- Como se predice la venta de las próximas 2 semanas, se puede calcular
el stock al final de la semana 1 y al conocer el forecast de demanda de la
semana 2, se puede saber cuánto pedir.
- Aplicando alguna regla para sobre-estimar ya que el costo de subestimar es mayor
al de sobreestimar
- Lo más importante es tener un forecast lo más preciso posible
"""

import pandas as pd

from utils.utils import read_processed_data, set_root_path


# set root repo
set_root_path()


""" 1. read "processed" data """
# leer archivos "processed" generados en step anterior
data, data_state, data_in_stock, data_master, data_submission = (
    read_processed_data(week_index="0", date_index="2024-04-08")
)


""" 2. read forecast generados """
folder_fcst = "data/submission/fcst"

data_fcst_output_train = pd.read_parquet(
    f"{folder_fcst}/data_fcst_output_train.parquet"
)
data_fcst_output_test = pd.read_parquet(
    f"{folder_fcst}/data_fcst_output_test.parquet"
)


####
data.head()


data_fcst_output_train.head()
