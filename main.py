import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

H, M, MIN_TRAIN = 3, 12, 36
MESES =  ['2025-06', '2025-07', '2025-08']

# CARGAR DATOS: 
# Una columna por SKU
# Una fila por MES
df = pd.concat(pd.read_excel(f) for f in glob.glob('*.xlsx'))
df['Fecha'] = pd.to_datetime(df['Fecha'])
piv = df.pivot(index='Fecha', columns='SKU', values='VentasMensuales').astype(float)
print(piv)
print("--------------------------------")

# Estacionalidad Compartida:
# Cada serie / su media, promediado por mes calendario

