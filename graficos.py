"""Funciones de gráficos para el pronóstico """
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

GRIS, AZUL, NARANJA = '#6b7280', '#2563eb', '#ea580c'

# En el gráfico de pronóstico, la línea gris es el histórico y la azul el pronóstico del método ganador de cada SKU; la banda sombreada es el intervalo empírico p10-p90, donde se espera que caiga el 80% de los valores reales. 
def grafico_pronostico(piv, salida, archivo='grafico_pronostico.png'):
    """Histórico + pronóstico con banda p10-p90, un panel por SKU (cada uno con su escala)."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axs = axs.ravel()
    for ax, sku in zip(axs, piv):
        s, f = piv[sku], salida[salida.SKU == sku]
        fechas = pd.to_datetime(f.mes)
        ax.plot(s.index, s, color=GRIS, lw=1.5, label='Histórico')
        ax.plot([s.index[-1], *fechas], [s.iloc[-1], *f.pronostico], color=AZUL, lw=2,
                marker='o', ms=4, label='Pronóstico')
        ax.fill_between(fechas, f.p10, f.p90, color=AZUL, alpha=0.2, label='Intervalo 80%')
        ax.set_title(f'{sku}  ·  {f.metodo.iloc[0]}', loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.25)
    axs[-1].axis('off')
    axs[-1].legend(*axs[0].get_legend_handles_labels(), loc='center', frameon=False)
    fig.suptitle('Ventas mensuales y pronóstico jun-ago 2025', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)

# En el gráfico de MASE, las celdas más oscuras indican peor desempeño y el recuadro negro marca el mejor método de cada SKU.
def grafico_mase(mase, archivo='grafico_mase.png'):
    """MASE por método y SKU (más oscuro = peor); borde = mejor método de cada SKU."""
    tabla = mase.assign(PROM=mase.mean(axis=1)).sort_values('PROM').drop(columns='PROM')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(tabla.values, cmap='Blues', vmin=0, vmax=1.6, aspect='auto')
    ax.set_xticks(range(tabla.shape[1]), tabla.columns)
    ax.set_yticks(range(tabla.shape[0]), tabla.index)
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.iat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white' if v > 1 else 'black')
    for j, sku in enumerate(tabla.columns):
        i = tabla.index.get_loc(tabla[sku].idxmin())
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='black', lw=2))
    ax.set_title('MASE del backtest (menor es mejor, 1.0 = naive estacional)', loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)

# Animación del backtest: cada cuadro es un corte. La línea gris oscura es lo que el modelo "conoce" hasta ese corte,
# la gris clara es el futuro que todavía no vio. La línea azul es lo que pronosticó para los H meses siguientes y los
# puntos naranjas lo que realmente se vendió; las rayas rojas entre ambos son el error. Los pronósticos de cortes
# anteriores quedan marcados en azul tenue, así se ve cómo se va armando el backtest.
def grafico_animacion_backtest(piv, r, mase, archivo='animacion_backtest.gif', intervalo=2000):
    """Anima el backtest corte por corte con el método ganador de cada SKU. Devuelve la animación."""
    mejor = mase.idxmin(axis=0)          # método con menor MASE para cada SKU
    cortes = sorted(r.corte.unique())
    H = int(r.paso.max())
    skus = list(piv.columns)

    # -> El backtest guarda err = real - pronóstico, así que el pronóstico se reconstruye como real - err (sin reentrenar).
    fcs = {}
    for sku in skus:
        g = r[(r.SKU == sku) & (r.metodo == mejor[sku])]
        for corte, gc in g.groupby('corte'):
            gc = gc.sort_values('paso')
            idx = [corte + p - 1 for p in gc.paso]
            real = piv[sku].values[idx]
            fcs[(sku, corte)] = (piv.index[idx], real - gc.err.values, real)

    # -> Límites fijos por SKU para que el eje no salte entre cuadros.
    ylim = {sku: 1.1 * max(piv[sku].max(), max(fcs[(sku, c)][1].max() for c in cortes)) for sku in skus}

    fig, axs = plt.subplots(3, 2, figsize=(12, 9))
    axs = axs.ravel()
    axs[-1].axis('off')
    axs[-1].legend(handles=[
        Line2D([], [], color=GRIS, lw=1.5, label='Historia conocida en el corte'),
        Line2D([], [], color=GRIS, lw=1, alpha=0.3, label='Futuro (el modelo no lo ve)'),
        Line2D([], [], color=AZUL, lw=2, marker='o', ms=4, label=f'Pronóstico a {H} meses'),
        Line2D([], [], color=NARANJA, marker='o', ls='', label='Real'),
        Line2D([], [], color='red', alpha=0.6, label='Error'),
        Line2D([], [], color=AZUL, lw=1, alpha=0.2, label='Pronósticos de cortes anteriores'),
    ], loc='center', frameon=False)

    def dibujar(i):
        corte = cortes[i]
        for ax, sku in zip(axs, skus):
            ax.clear()
            s = piv[sku]
            ax.plot(s.index, s, color=GRIS, lw=1, alpha=0.3)                       # serie completa, tenue
            ax.plot(s.index[:corte], s.iloc[:corte], color=GRIS, lw=1.5)          # lo conocido hasta el corte
            ax.axvline(s.index[corte - 1], color=GRIS, ls=':', lw=1)
            for c in cortes[:i]:                                                  # rastro de cortes previos
                f, fc, _ = fcs[(sku, c)]
                ax.plot(f, fc, color=AZUL, lw=1, alpha=0.2)
            f, fc, real = fcs[(sku, corte)]
            ax.plot([s.index[corte - 1], *f], [s.iloc[corte - 1], *fc], color=AZUL, lw=2, marker='o', ms=4)
            ax.vlines(f, fc, real, color='red', alpha=0.6, lw=1.5)
            ax.scatter(f, real, color=NARANJA, zorder=3, s=20)
            mae = abs(real - fc).mean()
            ax.set_title(f'{sku}  ·  {mejor[sku]}  ·  error medio de esta ventana: {mae:.1f}', loc='left', fontsize=10)
            ax.set_xlim(s.index[0], s.index[-1] + pd.DateOffset(months=1))   # margen para que no se corten los últimos puntos
            ax.set_ylim(0, ylim[sku])
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=0.25)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.suptitle(f'Backtest · corte {i + 1}/{len(cortes)}: entrena hasta {piv.index[corte - 1]:%Y-%m} '
                     f'({corte} meses) y pronostica los {H} siguientes', x=0.01, ha='left', fontweight='bold')

    # -> El último cuadro se repite unos ciclos para que la animación haga una pausa antes de volver a empezar.
    frames = list(range(len(cortes))) + [len(cortes) - 1] * 3
    dibujar(0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))   # deja lugar arriba para el título
    anim = FuncAnimation(fig, dibujar, frames=frames, interval=intervalo, repeat=True)
    anim.save(archivo, writer=PillowWriter(fps=1000 / intervalo))
    return anim
