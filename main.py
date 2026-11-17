import os
import sys
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

H = 3 # horizonte de pronóstico en meses (H), período estacional (M) y mínimo de meses de entrenamiento para el backtest (MIN_TRAIN)
M = 12 # período estacional: 12 meses por año 
MIN_TRAIN =36 # mínimo de meses de historia (3 años completos) antes de iniciar el backtest

AQUI = os.path.dirname(os.path.abspath(__file__))
# -> Las rutas se resuelven respecto del archivo y no del directorio desde el que se ejecuta.

# LOG: todo lo que se imprime va a la consola Y a un .txt, uno por corrida, para poder comparar corridas después.
class Tee:
    def __init__(self, *destinos):
        self.destinos = destinos
    def write(self, texto):
        for d in self.destinos:
            d.write(texto)
    def flush(self):
        for d in self.destinos:
            d.flush()

os.makedirs(os.path.join(AQUI, 'logs'), exist_ok=True)
LOG = os.path.join(AQUI, 'logs', datetime.now().strftime('log_%Y-%m-%d_%H%M%S.txt'))
sys.stdout = Tee(sys.__stdout__, open(LOG, 'w', encoding='utf-8'))
# -> Cada print escribe en los dos lados. El nombre lleva fecha y hora, así no se pisan las corridas.

def linea():
    return "--------------------------------"


# CARGAR DATOS:
# Una columna por SKU
# Una fila por MES
df = pd.concat(pd.read_excel(f) for f in glob.glob(os.path.join(AQUI, '*.xlsx')))
df['Fecha'] = pd.to_datetime(df['Fecha'])
piv = df.pivot(index='Fecha', columns='SKU', values='VentasMensuales').astype(float)
print(piv) # muestra la tabla de ventas: una fila por mes y una columna por SKU
print(linea())


MESES = pd.date_range(piv.index[-1], periods=H + 1, freq='MS')[1:].strftime('%Y-%m')
# -> Los meses a pronosticar salen del último dato disponible, no de una lista fija.


# ESTACIONALIDAD COMPARTIDA:
# Cada serie / su media, promediado por mes calendariow
# vuelve 12 números, uno por mes calendario, que dicen cuánto 
# se vende ese mes comparado con un mes promedio, juntando todos los SKU
def perfil_de(hist, grupo=None):
    # -> El perfil se estima a partir de los datos, así que dentro del backtest tiene 
    #    que construirse SOLO con la historia disponible hasta cada corte. 
    hist = hist if grupo is None else hist[grupo]
    norm = hist / hist.mean()
    # -> un índice relativo. 1.0 es un mes promedio para ese SKU y 1.3 es un 30 % por encima de su media
    # 1- Agrupar las filas por mes calendario sin importar el año
    # 2- Promediar cada grupo por columna (SKU) -> Para cada SKU tener su indice promedio de cada mes
    # 3- Promediar entre SKUs -> Un único perfil estacional compartido de 12 valores (uno por mes calendario)
    return norm.groupby(norm.index.month).mean().mean(axis=1)


def elegir_grupo(hist, min_corr=0.15):
    # -> Qué SKU comparten estacionalidad y merecen un perfil común. Se correlacionan los perfiles
    #    individuales y se conserva el que tiene correlación media positiva con el resto: una serie
    #    cuya estacionalidad va al revés que la del grupo no aporta señal, aporta sesgo.
    #    También se recalcula en cada corte, porque también es una decisión estimada con datos.
    norm = hist / hist.mean()
    perf = norm.groupby(norm.index.month).mean()
    c = perf.corr()
    media = (c.sum() - 1) / (len(c) - 1)
    grupo = media[media >= min_corr].index.tolist()
    return grupo if len(grupo) >= 2 else list(hist.columns)


# METODOS:
def metodos(y, mes, perfil, perfil_sel):
    hw = ExponentialSmoothing(y, trend='add', damped_trend=True, seasonal='add', seasonal_periods=M, initialization_method='estimated').fit().forecast(H)
    #- trend = true -> La tendencia es una cantidad fija de unidades por mes
    #- damped_trend = true -> agrega un parametro phi, que multiplica la tendencia en cada paso hacia adelante, evita que el pronostico se dispare o vaya a 0.
    #   En vez de proyectar +5, +5, +5… para siempre, proyecta +5·φ, +5·φ², +5·φ³…
    #- seasonal=add -> La estacionalidad es aditiva: cada mes calendario suma o resta una cantidad fija de unidades respecto al nivel base, sin importar qué tan alto o bajo esté ese nivel.
    #   Por ejemplo, si diciembre suma +20 unidades, lo hará igual con ventas de 100 que con ventas de 500 (a diferencia de la estacionalidad multiplicativa, que escala con el nivel).
    #- seasonal_periods=M -> El patrón estacional se repite cada 12 meses.
    #- initialization_method='estimated' -> El nivel, la tendencia y los factores estacionales iniciales no se fijan con una heurística, sino que se estiman junto con el resto de los parámetros durante el ajuste.
    #- .fit().forecast(H) -> Ajusta el modelo con la serie de entrenamiento y pronostica los próximos H meses.

    sarima = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, M), trend='c', enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(H)
    #- order=(1, 0, 0) -> Un término autorregresivo de corto plazo, sin diferenciación ni promedio móvil.
    #- seasonal_order=(1, 0, 0, M) -> Un término autorregresivo estacional con periodo de 12 meses.
    #- trend='c' -> Agrega una constante que captura el nivel base de la serie.
    #- enforce_stationarity=False, enforce_invertibility=False -> Evita restricciones que hacen fallar el ajuste con series cortas.
    #- .fit(disp=False).forecast(H) -> Ajusta el modelo sin imprimir el log de optimización y pronostica los próximos H meses.

    nivel = y[-M:].mean()
    meses_fc = [(mes + i) % 12 + 1 for i in range(H)]
    #- Mes calendario (1 a 12) de cada uno de los H meses a pronosticar, avanzando desde el mes siguiente al último dato.

    grupo = nivel * np.array([perfil[m] for m in meses_fc])
    #- Nivel base (promedio de los últimos 12 meses) * perfil -> estacionalidad multiplicativa compartida entre TODOS los SKU.

    grupo_sel = nivel * np.array([perfil_sel[m] for m in meses_fc])
    #- Igual, pero con el perfil armado solo con los SKU que comparten forma estacional.
    #  Comparar grupo contra grupo_sel aísla el efecto de CON QUIÉN se agrupa.

    media_12 = np.repeat(nivel, H)
    #- y[-M:].mean() -> Nivel base: promedio de los últimos 12 meses, estable frente a la estacionalidad y el ruido mensual.
    #- np.repeat(..., H) -> Repite ese valor constante para cada uno de los H meses a pronosticar.
    #  Comparar grupo contra media_12 aísla el efecto de APLICAR un perfil estacional: mismo nivel, con y sin estacionalidad.

    media_3 = np.repeat(y[-3:].mean(), H)
    #- y[-3:].mean() -> Nivel base: promedio de los últimos 3 meses, más sensible a cambios recientes que el de 12 meses.

    s_naive = y[-M:][:H]
    #- Repite los mismos meses calendario del año anterior. Es el baseline contra el que se escala el MASE.

    combinacion = np.mean([media_12, hw, grupo_sel, sarima], axis=0)

    #- aux -> Diccionario que guarda los pronósticos de cada modelo y de la combinación, para compararlos o evaluarlos después por separado.
    aux = {"hw": hw, "sarima": sarima, "grupo": grupo, "grupo_sel": grupo_sel,
           "media_3": media_3, "media_12": media_12, "s_naive": s_naive, "combinacion": combinacion}

    # No se pueden vender −3 unidades, y algunos métodos pueden dar negativos:
    return {k: np.clip(v, 0, None) for k, v in aux.items()}
    # - > np.clip([42.1, 38.5, -3.2], 0, None)  →  [42.1, 38.5, 0.0]

# BACKTEST: simulación del paso del tiempo
# ->  el mismo mes se pronostica varias veces, cada vez con más información.
filas = []
for corte in range(MIN_TRAIN, len(piv) - H + 1):
    # -> Recorre los puntos de corte posibles: cada corte usa los primeros `corte` meses como entrenamiento y los H meses siguientes como prueba.
    hist = piv.iloc[:corte]
    # -> El perfil y la elección del grupo se arman una sola vez por corte, con los datos de ese corte y de ningún mes posterior.
    perfil, grupo_sel = perfil_de(hist), elegir_grupo(hist)
    perfil_s = perfil_de(hist, grupo_sel)
    #   Es el mismo perfil de 12 números, pero armado solo con los SKU del grupo elegido. Así se pueden comparar los dos métodos: 
    #   grupo usa el perfil de todos los SKU y grupo_sel usa el perfil de los SKU parecidos.
    mes = piv.index[corte - 1].month
    for sku in piv:
        v = piv[sku].values
        tr, real = v[:corte], v[corte:corte + H]
        # -> tr es el historial disponible hasta el corte (lo que el modelo "conoce") y real son los H meses siguientes, que se usan para evaluar el pronóstico.
        escala = np.mean(np.abs(tr[M:] - tr[:-M]))      # error del naive estacional (para MASE)
        for met, fc in metodos(tr, mes, perfil, perfil_s).items():
            # -> Recorre cada método y su pronóstico fc de H meses; el MASE compara el error absoluto medio contra `real` y lo divide por la escala del naive estacional, de modo que un valor menor a 1 supera al naive.
            for paso in range(H):
                # -> err = real - pronóstico (CON SIGNO): el signo sirve para medir sesgo, el valor absoluto para medir magnitud.
                filas.append({"SKU": sku, "corte": corte, "metodo": met, "paso": paso + 1,
                              "err": real[paso] - fc[paso], "escala": escala})
# -> Convierte la lista de filas en un DataFrame para poder agrupar y comparar los métodos. Se conserva `corte` para poder mirar la dispersión entre ventanas.
r = pd.DataFrame(filas, columns=['SKU', 'corte', 'metodo', 'paso', 'err', 'escala'])
r['abs'] = r.err.abs()
r['mase'] = r['abs'] / r.escala
cortes = r.corte.nunique()
print(f'Backtest: {cortes} orígenes x {H} pasos x {piv.shape[1]} SKU = {cortes*H*piv.shape[1]} pronósticos por método')
print(linea())

# Para cada celda (método, SKU) promedia los 45 MASE de las 15 ventanas × 3 pasos 
# -> Un solo método para los 3 meses.
mase = r.pivot_table(index='metodo', columns='SKU', values='mase')
print('=== MASE por método (menor es mejor; 1.0 = naive estacional) ===')
print(mase.assign(PROMEDIO=mase.mean(axis=1)).sort_values('PROMEDIO').round(3), '\n')

# -> El MASE es adimensional y comparable entre SKU, pero un MASE bajo también puede significar que
#    el baseline es malo. Por eso se reporta al lado el numerador en unidades.
mae = r.pivot_table(index='metodo', columns='SKU', values='abs')
print('=== MAE en unidades (el numerador del MASE, para saber si el modelo es bueno o el baseline es malo) ===')
print(mae.round(1), '\n')
print('escala del naive estacional por SKU (el denominador):')
print(r.groupby('SKU').escala.mean().round(1).to_string(), '\n')

# -> El error casi siempre crece con el horizonte. La pendiente dice hasta dónde el modelo sirve.
print('=== MASE por horizonte (promedio de los 5 SKU) ===')
print(r.pivot_table(index='metodo', columns='paso', values='mase').round(3), '\n')

# Por qué 15 orígenes y no 1: el error de una sola ventana es casi ruido.
# -> Con una sola ventana la medición podía caer en cualquier punto de este rango
mejor_global = mase.mean(axis=1).idxmin()
disp = r[r.metodo == mejor_global].groupby('corte')['abs'].mean()
print(f'=== Dispersión entre ventanas ({mejor_global}, error absoluto medio por origen) ===')
print(f'min {disp.min():.1f} | max {disp.max():.1f} | promedio {disp.mean():.1f} | desvío {disp.std():.1f}')

# -> Error CON SIGNO: mide sesgo. Dos modelos pueden tener el mismo MAE y uno quedarse corto siempre.
#    En inventario eso no se compensa: se acumula y se come el stock de seguridad.
print('=== Sesgo: error medio con signo (positivo = el modelo pronostica de menos) ===')
print(r.pivot_table(index='metodo', columns='SKU', values='err').round(1), '\n')


# AGRUPAMIENTO: con quién conviene compartir estacionalidad
norm_full = piv / piv.mean()
corr = norm_full.groupby(norm_full.index.month).mean().corr()
print('=== Correlación entre perfiles estacionales ===')
print(corr.round(2).to_string())
print('\ncorrelación media de cada SKU con el resto:')
print(((corr.sum() - 1) / (len(corr) - 1)).sort_values(ascending=False).round(2).to_string())
print(f'\ngrupo elegido en el último corte: {elegir_grupo(piv)}')
print(linea())

# ERROR ACUMULADO SOBRE EL HORIZONTE
# -> Para dimensionar stock no importa el error de un mes sino el del acumulado de los H meses que
#    dura el horizonte de riesgo. No es H veces el error mensual ni raíz(H) veces: los errores están
#    correlacionados. Se mide directo del backtest en vez de suponer independencia.
acum = r[r.metodo == mejor_global].groupby(['SKU', 'corte']).err.sum().abs().groupby('SKU').mean()
mensual = r[r.metodo == mejor_global].groupby('SKU')['abs'].mean()
comp = pd.DataFrame({'error mensual': mensual.round(1), 'error acumulado 3m': acum.round(1),
                     'ratio': (acum / mensual).round(2)})
print(f'=== Error del acumulado de {H} meses ({mejor_global}) ===')
print(comp.to_string())
print(f'Si los errores fueran independientes el ratio sería raíz({H}) = {np.sqrt(H):.2f}; '
      f'si fueran idénticos sería {H}.\n')

# -> Tracking signal: suma de los errores con signo de los últimos K orígenes, dividida por el MAD 
#    (Desviación Absoluta Media de esos mismos orígenes).
#    Se usa SOLO el paso 1, porque es el único horizonte con un pronóstico por mes: si se mezclaran
#    los tres pasos, cada mes entraría tres veces y la suma se inflaría sin que haya más evidencia.
#    Se divide por el MAD para que el umbral valga igual en un SKU de 40 unidades/mes y en uno de 180.
K = 6
h1 = r[(r.metodo == mejor_global) & (r.paso == 1)].sort_values('corte')
ts = h1.groupby('SKU').apply(lambda g: g.err.tail(K).sum() / g['abs'].tail(K).mean()).round(2)
print(f'=== Tracking signal ({mejor_global}, paso 1, últimos {K} orígenes) ===')
print(ts.to_string())
alarma = ts[ts.abs() > 4].index.tolist()
print('Umbral |TS| > 4:', ('ALARMA en ' + ', '.join(alarma)) if alarma else 'ningún SKU lo supera.')
print()

# PRONOSTICOS FINALES
perfil_f, grupo_f = perfil_de(piv), elegir_grupo(piv)
perfil_fs = perfil_de(piv, grupo_f)
salida = []
for sku, met in mase.idxmin(axis=0).items():
    # -> Para cada SKU toma el método con menor MASE promedio (met) y lo reentrena con todo el historial disponible para generar el pronóstico de los H meses futuros.
    fc = metodos(piv[sku].values, piv.index[-1].month, perfil_f, perfil_fs)[met]
    for paso, (mes, val) in enumerate(zip(MESES, fc), 1):
        e = r[(r.SKU == sku) & (r.metodo == met) & (r.paso == paso)].err
        # -> Suma al pronóstico los percentiles 10 y 90 de los errores históricos de ese método, SKU y paso (e) para construir un intervalo empírico (del 80%); el límite inferior se trunca en 0 porque no puede haber ventas negativas.
        salida.append({'fecha_corrida': piv.index[-1].strftime('%Y-%m'),
                       'SKU': sku, 'metodo': met, 'mes': mes, 'paso': paso, 'pronostico': round(val),
                       'p10': max(0, round(val + np.percentile(e, 10))),
                       'p90': round(val + np.percentile(e, 90))})

salida = pd.DataFrame(salida)
print('=== Pronóstico (intervalo 80 % empírico, de los errores del backtest) ===')
print(salida.to_string(index=False))

# -> Se guardan los resultados para que la corrida sea auditable y reproducible.
salida.to_csv(os.path.join(AQUI, 'pronostico.csv'), index=False)
r.to_csv(os.path.join(AQUI, 'backtest.csv'), index=False)
print('\nGuardado en pronostico.csv y backtest.csv')
print(f'Log de esta corrida: {LOG}')

# Gráficos
from graficos import grafico_pronostico, grafico_mase, grafico_animacion_backtest
grafico_pronostico(piv, salida)
grafico_mase(mase)
anim = grafico_animacion_backtest(piv, r, mase)  # -> guardar la referencia: si no, Python la borra y la animación no corre
plt.show()