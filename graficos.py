"""Funciones de gráficos para el pronóstico """
import pandas as pd
import matplotlib.pyplot as plt

GRIS, AZUL = '#6b7280', '#2563eb'

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