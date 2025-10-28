# %%
"""
Actualizar data raw
Luego de cargar la data RAW de la semana 'i', en los folders correspondientes
Correr el siguiente script para crear data raw en folder "data/input/to_models"
para generar archivos de la semana 'i' en el formato que se necesita para correr
todos los modelos
"""

from utils.utils import update_raw_to_models


# correr si es la primera vez
# from utils.utils import init_raw_to_models_week0
# init_raw_to_models_week0()

# correr para el resto de las semanas
current_week = 2
update_raw_to_models(week_index=current_week)
