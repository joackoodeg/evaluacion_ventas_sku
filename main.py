import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

H, M, MIN_TRAIN = 3, 12, 36
MESES =  ['2025-06', '2025-07', '2025-08']
def linea():
    return "--------------------------------"


# CARGAR DATOS: 
# Una columna por SKU
# Una fila por MES
df = pd.concat(pd.read_excel(f) for f in glob.glob('*.xlsx'))
df['Fecha'] = pd.to_datetime(df['Fecha'])
piv = df.pivot(index='Fecha', columns='SKU', values='VentasMensuales').astype(float)
print(piv)
print(linea())

# Estacionalidad Compartida:
# Cada serie / su media, promediado por mes calendario
norm = piv / piv.mean()
print(norm)
print(linea())
# un índice relativo. 1.0 es un mes promedio para ese SKU y 1.3 es un 30 % por encima de su media

# Perfil Estacional:
# 1- Agrupar las filas por mes calendario sin importar el año
# 2- Promediar cada grupo por columna (SKU) -> Para cada SKU tener su indice promedio de cada mes
# 3- Promediar entre SKUs -> Un único perfil estacional compartido de 12 valores (uno por mes calendario)
perfil = norm.groupby(norm.index.month).mean().mean(axis=1)
print(perfil)
print(linea())

# METODOS:
def metodos(y, mes):
    hw = ExponentialSmoothing(y, trend='add', damped_trend=True, seasonal='add', seasonal_periods=M, initialization_method='estimated').fit().forecast(H)
    #- trend = true -> La tendencia es una cantidad fija de unidades por mes
    #- damped_trend = true -> agrega un parametro phi, que multiplica la tendencia en cada paso hacia adelante, evita que el pronostico se dispare o vaya a 0.
    #   En vez de proyectar +5, +5, +5… para siempre, proyecta +5·φ, +5·φ², +5·φ³…
    #- seasonal=add -> La estacionalidad es aditiva: cada mes calendario suma o resta una cantidad fija de unidades respecto al nivel base, sin importar qué tan alto o bajo esté ese nivel.
    #   Por ejemplo, si diciembre suma +20 unidades, lo hará igual con ventas de 100 que con ventas de 500 (a diferencia de la estacionalidad multiplicativa, que escala con el nivel).
    #- seasonal_periods=M -> El patrón estacional se repite cada 12 meses.
    #- initialization_method='estimated' -> El nivel, la tendencia y los factores estacionales iniciales no se fijan con una heurística, sino que se estiman junto con el resto de los parámetros durante el ajuste.
    #- .fit().forecast(H) -> Ajusta el modelo con la serie de entrenamiento y pronostica los próximos H meses.

    sarima = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, M), trend='c',enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(H)
    #- order=(1, 0, 0) -> Un término autorregresivo de corto plazo, sin diferenciación ni promedio móvil.
    #- seasonal_order=(1, 0, 0, M) -> Un término autorregresivo estacional con periodo de 12 meses.
    #- trend='c' -> Agrega una constante que captura el nivel base de la serie.
    #- enforce_stationarity=False, enforce_invertibility=False -> Evita restricciones que hacen fallar el ajuste con series cortas.
    #- .fit(disp=False).forecast(H) -> Ajusta el modelo sin imprimir el log de optimización y pronostica los próximos H meses.

    grupo =  y[-M:].mean() * np.array([perfil[(mes + i) % 12 + 1] for i in range(H)])
    #- y[-M:].mean() -> Nivel base: promedio de los últimos 12 meses.
    #- perfil[(mes + i) % 12 + 1] -> Índice estacional del mes calendario (1 a 12) de cada uno de los H meses a pronosticar, avanzando desde el mes siguiente al último dato.
    #- Nivel base * perfil -> Pronóstico con estacionalidad multiplicativa compartida entre SKUs.

    media_12 = np.repeat(y[-M:].mean(),H)
    #- y[-M:].mean() -> Nivel base: promedio de los últimos 12 meses, estable frente a la estacionalidad y el ruido mensual.
    #- np.repeat(..., H) -> Repite ese valor constante para cada uno de los H meses a pronosticar.

    
    media_3 = np.repeat(y[-3:].mean(),H)
    #- y[-3:].mean() -> Nivel base: promedio de los últimos 3 meses, más sensible a cambios recientes que el de 12 meses.
    #- np.repeat(..., H) -> Repite ese valor constante para cada uno de los H meses a pronosticar.

    s_naive = y[-M:][:H]

    combinacion = np.mean([media_12, hw, grupo, sarima], axis=0)
    
    #- aux -> Diccionario que guarda los pronósticos de cada modelo y de la combinación, para compararlos o evaluarlos después por separado.
    aux = {"hw": hw, "sarima": sarima, "grupo": grupo, "media_3": media_3, "media_12": media_12, "s_naive": s_naive, "combinacion": combinacion}

    return {k: np.clip(v, 0, None) for k, v in aux.items()}

#- Ejemplo de uso: se elige un SKU y un método para imprimir su pronóstico a H meses, partiendo del último mes disponible en el pivote.
sku = piv.columns[0]
met = "combinacion"
print(metodos(piv[sku].values, piv.index[-1].month)[met])