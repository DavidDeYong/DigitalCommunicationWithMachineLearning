"""
graficas/graficas.py
====================
Visualizaciones del benchmark unificado.

Figuras generadas:
  1. curva_ber_principal()          — BER vs Eb/N0 (escala log completa)
  2. curva_ber_zoom()               — BER vs Eb/N0 (zoom región útil)
  3. grafica_degradacion()          — degradación en dB respecto a Bayes
  4. tabla_metricas()               — tabla visual de métricas comparativas
  5. regiones_decision()            — regiones de decisión en plano IQ
  6. historial_loss()               — curvas de pérdida de redes neuronales
  7. histograma_simbolos()          — histograma completo pre/post scrambling
  8. histograma_articulo_pre()      — histograma minimalista PRE scrambling (artículo)
  9. histograma_articulo_post()     — histograma minimalista POST scrambling (artículo)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

from canal.canal_awgn import ber_teorica_16qam


# ---------------------------------------------------------------------------
# Paleta de colores y marcadores
# ---------------------------------------------------------------------------
_COLORES = {
    'Bayes'         : '#2196F3',
    'SVM RBF'       : '#E91E63',
    'SVM Lineal'    : '#FF9800',
    'KNN'           : '#4CAF50',
    'Random Forest' : '#9C27B0',
    'MLP'           : '#00BCD4',
    'Red Profunda'  : '#F44336',
    'Teorica'       : '#AAAAAA',   # ← gris claro (referencia discreta)
}

_MARKERS = {
    'Bayes'         : 'o',
    'SVM RBF'       : 's',
    'SVM Lineal'    : '^',
    'KNN'           : 'D',
    'Random Forest' : 'P',
    'MLP'           : 'X',
    'Red Profunda'  : '*',
}

def _color(nombre):
    for k, v in _COLORES.items():
        if k.lower() in nombre.lower():
            return v
    return '#607D8B'

def _marker(nombre):
    for k, v in _MARKERS.items():
        if k.lower() in nombre.lower():
            return v
    return 'o'

def _guardar(fig, nombre_archivo, dir_resultados):
    os.makedirs(dir_resultados, exist_ok=True)
    ruta = os.path.join(dir_resultados, nombre_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"  [Graficas] Guardado: '{ruta}'")

def _separar_confiable(reg):
    """Devuelve (x_conf, y_conf, x_noconf, y_noconf) para un registro."""
    x = np.array(reg['Eb_N0_dB'])
    y = np.array(reg['ber'], dtype=float)
    c = np.array(reg.get('confiable', [True]*len(x)), dtype=bool)
    y_plot = np.where(y > 0, y, np.nan)
    return x[c], y_plot[c], x[~c], y_plot[~c]


# ---------------------------------------------------------------------------
# Función interna compartida para graficar una curva BER
# ---------------------------------------------------------------------------
def _graficar_ber_en_ax(ax, registros, Eb_N0_fino, ber_teo_fino):
    """Dibuja la curva teórica y todas las curvas de clasificadores en un eje."""
    # Curva teórica: gris claro, línea más delgada → actúa como referencia discreta
    ax.semilogy(Eb_N0_fino, ber_teo_fino,
                color=_COLORES['Teorica'], linewidth=1.8,
                linestyle='-', label='BER Teórica 16-QAM', zorder=5)

    for reg in registros:
        color  = _color(reg['nombre'])
        marker = _marker(reg['nombre'])
        xc, yc, xn, yn = _separar_confiable(reg)

        if len(xc) > 0:
            ax.semilogy(xc, yc, marker=marker, linestyle='--',
                        color=color, linewidth=1.5, markersize=7,
                        label=reg['nombre'], zorder=6)
        if len(xn) > 0:
            lbl = f"{reg['nombre']} (insuf.)" if len(xc) == 0 else "_nolegend_"
            ax.semilogy(xn, yn, marker=marker, linestyle=':',
                        color=color, linewidth=1.0, markersize=9,
                        markerfacecolor='none', markeredgewidth=1.5,
                        label=lbl, zorder=6)

    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=12)
    ax.set_ylabel(r'Probabilidad de error de bit $P_b$', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='lower left')


# ---------------------------------------------------------------------------
# 1. Curva BER principal (escala completa)
# ---------------------------------------------------------------------------
def curva_ber_principal(registros, guardar=False, dir_resultados="resultados/"):
    """BER vs Eb/N0 — figura completa, escala logarítmica."""
    Eb_N0_min  = min(r['Eb_N0_dB'][0]  for r in registros if r['Eb_N0_dB'])
    Eb_N0_max  = max(r['Eb_N0_dB'][-1] for r in registros if r['Eb_N0_dB'])
    Eb_N0_fino = np.linspace(Eb_N0_min - 1, Eb_N0_max + 1, 400)
    ber_teo    = ber_teorica_16qam(Eb_N0_fino)

    fig, ax = plt.subplots(figsize=(9, 6))
    _graficar_ber_en_ax(ax, registros, Eb_N0_fino, ber_teo)
    ax.set_title('BER vs $E_b/N_0$ — Clasificadores ML para 16-QAM\n'
                 'Canal AWGN — Protocolo de entrenamiento realista', fontsize=12)
    fig.text(0.5, -0.02,
             'Marcador vacío (···): estadística insuficiente (<100 errores).',
             ha='center', fontsize=8.5, color='dimgray', style='italic')
    plt.tight_layout()
    if guardar:
        _guardar(fig, 'ber_curva_completa.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 2. Curva BER zoom (región útil)
# ---------------------------------------------------------------------------
def curva_ber_zoom(registros, guardar=False, dir_resultados="resultados/"):
    """BER vs Eb/N0 — zoom en la región de interés (BER 10^-1 a 10^-5)."""
    Eb_N0_min  = min(r['Eb_N0_dB'][0]  for r in registros if r['Eb_N0_dB'])
    Eb_N0_max  = max(r['Eb_N0_dB'][-1] for r in registros if r['Eb_N0_dB'])
    Eb_N0_fino = np.linspace(Eb_N0_min - 1, Eb_N0_max + 1, 400)
    ber_teo    = ber_teorica_16qam(Eb_N0_fino)

    fig, ax = plt.subplots(figsize=(9, 6))
    _graficar_ber_en_ax(ax, registros, Eb_N0_fino, ber_teo)
    ax.set_ylim([1e-5, 5e-1])
    ax.set_xlim([Eb_N0_min - 0.5, Eb_N0_max + 0.5])
    ax.set_title('BER vs $E_b/N_0$ — Zoom región de interés\n'
                 'Canal AWGN — Protocolo de entrenamiento realista', fontsize=12)
    fig.text(0.5, -0.02,
             'Marcador vacío (···): estadística insuficiente (<100 errores).',
             ha='center', fontsize=8.5, color='dimgray', style='italic')
    plt.tight_layout()
    if guardar:
        _guardar(fig, 'ber_curva_zoom.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Gráfica de degradación
# ---------------------------------------------------------------------------
def grafica_degradacion(registros, guardar=False, dir_resultados="resultados/"):
    """
    Degradación en dB respecto al detector Bayesiano.

    Para cada clasificador ML y cada nivel de BER, calcula cuántos dB
    adicionales necesita respecto a Bayes para alcanzar esa misma BER.

    Degradacion(BER) = Eb/N0_clasificador(BER) - Eb/N0_Bayes(BER)

    Un valor de 0 dB significa rendimiento igual al Bayesiano.
    Un valor positivo indica cuántos dB extra se necesitan (peor rendimiento).
    """
    Eb_N0_denso = np.linspace(-2, 20, 5000)
    ber_teo_den = ber_teorica_16qam(Eb_N0_denso)

    reg_bayes = next((r for r in registros
                      if 'bayes' in r['nombre'].lower()), None)
    if reg_bayes is None:
        print("  [Graficas] No se encontró el clasificador Bayes. "
              "Gráfica de degradación omitida.")
        return

    x_b = np.array(reg_bayes['Eb_N0_dB'])
    y_b = np.array(reg_bayes['ber'], dtype=float)
    c_b = np.array(reg_bayes.get('confiable', [True]*len(x_b)), dtype=bool)

    ber_validas = y_b[c_b & (y_b > 0)]
    if len(ber_validas) < 2:
        print("  [Graficas] Datos insuficientes para gráfica de degradación.")
        return

    ber_min     = max(ber_validas.min(), 1e-6)
    ber_max     = ber_validas.max()
    ber_targets = np.logspace(np.log10(ber_min), np.log10(ber_max), 200)

    try:
        interp_teo = interp1d(ber_teo_den[::-1], Eb_N0_denso[::-1],
                              kind='linear', bounds_error=False,
                              fill_value=(Eb_N0_denso[-1], Eb_N0_denso[0]))
        EbN0_bayes_teo = interp_teo(ber_targets)
    except Exception:
        print("  [Graficas] Error en interpolación para degradación.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.axhline(0, color=_COLORES['Teorica'], linewidth=1.8,
               linestyle='-', label='Bayes teórico (referencia 0 dB)', zorder=10)

    clf_sin_bayes = [r for r in registros if 'bayes' not in r['nombre'].lower()]

    for reg in clf_sin_bayes:
        x   = np.array(reg['Eb_N0_dB'])
        y   = np.array(reg['ber'], dtype=float)
        c   = np.array(reg.get('confiable', [True]*len(x)), dtype=bool)
        color  = _color(reg['nombre'])
        marker = _marker(reg['nombre'])

        mask = c & (y > 0)
        if mask.sum() < 2:
            continue

        x_ok = x[mask]
        y_ok = y[mask]

        try:
            interp_clf = interp1d(y_ok[::-1], x_ok[::-1],
                                  kind='linear', bounds_error=False,
                                  fill_value=np.nan)
            EbN0_clf = interp_clf(ber_targets)
        except Exception:
            continue

        degradacion = EbN0_clf - EbN0_bayes_teo

        mask_plot = np.isfinite(degradacion) & \
                    (ber_targets >= y_ok.min() * 0.9) & \
                    (ber_targets <= y_ok.max() * 1.1)

        if mask_plot.sum() < 2:
            continue

        ax.plot(ber_targets[mask_plot], degradacion[mask_plot],
                color=color, linewidth=1.8, label=reg['nombre'])

        idx_max = np.argmax(np.abs(degradacion[mask_plot]))
        bx = ber_targets[mask_plot][idx_max]
        dy = degradacion[mask_plot][idx_max]
        ax.plot(bx, dy, marker=marker, color=color, markersize=8, zorder=5)

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel(r'BER objetivo ($P_b$)', fontsize=12)
    ax.set_ylabel(r'Degradación respecto a Bayes (dB)', fontsize=12)
    ax.set_title('Degradación en $E_b/N_0$ respecto al detector Bayesiano\n'
                 'Positivo = más dB necesarios para la misma BER', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3,
               alpha=0.04, color='red', label='_nolegend_')

    fig.text(0.5, -0.02,
             'Degradación = Eb/N0 necesario por el clasificador ML menos Eb/N0 del Bayes teórico '
             'para la misma BER.\n'
             '0 dB = rendimiento idéntico al óptimo. '
             'Valores mayores indican mayor consumo de potencia para igual calidad.',
             ha='center', fontsize=8.5, color='dimgray', style='italic')
    plt.tight_layout()
    if guardar:
        _guardar(fig, 'degradacion_vs_bayes.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 4. Tabla de métricas
# ---------------------------------------------------------------------------
def tabla_metricas(registros, guardar=False, dir_resultados="resultados/"):
    nombres   = [r['nombre'] for r in registros]
    ber_media = [np.nanmean(r['ber'])            for r in registros]
    acc_media = [np.mean(r['accuracy'])           for r in registros]
    t_train   = [np.mean(r['t_entrenamiento_s']) for r in registros]
    t_inf     = [np.mean(r['t_inferencia_us'])   for r in registros]

    fig, ax = plt.subplots(figsize=(13, len(nombres) * 0.75 + 2))
    ax.axis('off')

    columnas = ['Clasificador', 'BER media', 'Accuracy\nmedia',
                'T. entrenamiento\n(s, promedio)', 'T. inferencia\n(µs/símbolo)']
    filas = []
    for i, nombre in enumerate(nombres):
        ber_str = f"{ber_media[i]:.3e}" if ber_media[i] > 0 else "< 10-5"
        filas.append([nombre, ber_str, f"{acc_media[i]*100:.2f}%",
                      f"{t_train[i]:.2f}", f"{t_inf[i]:.4f}"])

    tabla = ax.table(cellText=filas, colLabels=columnas,
                     loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    for j in range(len(columnas)):
        tabla[0, j].set_facecolor('#1976D2')
        tabla[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(filas) + 1):
        color = '#E3F2FD' if i % 2 == 0 else 'white'
        for j in range(len(columnas)):
            tabla[i, j].set_facecolor(color)

    plt.title('Tabla comparativa de clasificadores — Benchmark 16-QAM',
              fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    if guardar:
        _guardar(fig, 'tabla_metricas.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Regiones de decisión en el plano IQ
# ---------------------------------------------------------------------------
def regiones_decision(clasificadores, Eb_N0_dB, A=1.0,
                      guardar=False, dir_resultados="resultados/"):
    from canal.modulador import _TODOS_IQ

    n_clf = len(clasificadores)
    cols  = min(3, n_clf)
    rows  = (n_clf + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = np.array(axes).flatten() if n_clf > 1 else [axes]

    margen = 0.5
    lim    = A * 3 + margen
    paso   = A * 0.05
    xx, yy = np.meshgrid(np.arange(-lim, lim + paso, paso),
                         np.arange(-lim, lim + paso, paso))
    malla  = np.c_[xx.ravel(), yy.ravel()]
    cmap16 = plt.cm.get_cmap('tab20', 16)
    constelacion = A * _TODOS_IQ

    for i, clf in enumerate(clasificadores):
        ax = axes[i]
        try:
            Z = clf.predict(malla).reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 16.5, 1),
                        cmap=cmap16, alpha=0.4)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha='center', va='center')
        ax.scatter(constelacion[:, 0], constelacion[:, 1],
                   c='black', s=80, zorder=5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('I', fontsize=11); ax.set_ylabel('Q', fontsize=11)
        ax.set_title(f'{clf.nombre}\n$E_b/N_0$ = {Eb_N0_dB} dB', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Regiones de decisión — $E_b/N_0$ = {Eb_N0_dB} dB',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if guardar:
        _guardar(fig, f'regiones_decision_EbN0_{Eb_N0_dB}dB.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 6. Historial de pérdida de redes neuronales
# ---------------------------------------------------------------------------
def historial_loss(redes, Eb_N0_dB, guardar=False, dir_resultados="resultados/"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for red in redes:
        if hasattr(red, '_historial_loss') and red._historial_loss:
            ax.plot(red._historial_loss, label=red.nombre,
                    color=_color(red.nombre), linewidth=1.8)
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title(f'Convergencia del entrenamiento — $E_b/N_0$ = {Eb_N0_dB} dB',
                 fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if guardar:
        _guardar(fig, f'loss_redes_EbN0_{Eb_N0_dB}dB.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Histograma de símbolos transmitidos (versión completa — diagnóstico)
# ---------------------------------------------------------------------------
def histograma_simbolos(fuente_dict, A=1.0, n_simbolos=50_000,
                        usar_scrambling=False, scrambling_seed=42,
                        scrambling_poly=0x400007,
                        guardar=False, dir_resultados="resultados/"):
    """
    Analiza la distribución de los símbolos 16-QAM transmitidos.

    Genera SIEMPRE la figura con la distribución ANTES del scrambling
    (fuente de voz cruda). Si usar_scrambling=True, genera además la
    figura DESPUÉS del scrambling (condición real del benchmark).
    """
    from canal.modulador import modular, simbolos_a_etiquetas, _TODOS_IQ
    from fuente.fuente   import extraer_segmento
    from scipy.stats     import chisquare

    n_sim = min(n_simbolos, fuente_dict['n_simbolos'])

    def _analizar(bits):
        etiq     = simbolos_a_etiquetas(modular(bits, A=A), A=A)
        counts   = np.bincount(etiq, minlength=16)
        freqs    = counts / counts.sum()
        freq_uni = 1.0 / 16
        desvios  = (freqs - freq_uni) / freq_uni * 100
        p1_bits  = bits[:n_sim * 4].reshape(-1, 4).mean(axis=0)
        chi2, pval = chisquare(counts)
        return freqs, desvios, counts, p1_bits, chi2, pval

    # ── Figura A: fuente cruda (sin scrambling) ──────────────────────────────
    bits_crudo = extraer_segmento(fuente_dict, 0, n_sim, usar_scrambling=False)
    _histograma_articulo(
        bits           = bits_crudo,
        n_sim          = n_sim,
        A              = A,
        titulo         = 'Distribución de símbolos 16-QAM — sin scrambling',
        nombre_archivo = 'histograma_sin_scrambling.png',
        guardar        = guardar,
        dir_resultados = dir_resultados,
    )

    # ── Figura B: post-scrambling ─────────────────────────────────────────
    if usar_scrambling:
        bits_scram = extraer_segmento(fuente_dict, 0, n_sim,
                                      usar_scrambling=True,
                                      scrambling_seed=scrambling_seed,
                                      scrambling_poly=scrambling_poly)
        _histograma_articulo(
            bits           = bits_scram,
            n_sim          = n_sim,
            A              = A,
            titulo         = 'Distribución de símbolos 16-QAM — con scrambling PRBS-23',
            nombre_archivo = 'histograma_con_scrambling.png',
            guardar        = guardar,
            dir_resultados = dir_resultados,
        )
    else:
        print("\n  [Histograma] Solo se muestra la distribución de la fuente cruda.")
        print("  [Histograma] Activar USAR_SCRAMBLING=True en config.py para ver "
              "la comparación completa antes/después.")


# ---------------------------------------------------------------------------
# Función interna compartida: histograma minimalista para artículo
# ---------------------------------------------------------------------------
def _histograma_articulo(bits, n_sim, A, titulo, nombre_archivo,
                         guardar, dir_resultados):
    """
    Histograma de frecuencia de símbolos 16-QAM para publicación en artículo.

    Diseño minimalista:
      - Barras de frecuencia relativa por símbolo (0–15)
      - Línea de referencia equiprobable en gris punteado
      - Banda de tolerancia ±5% alrededor de la equiprobable
      - Test chi-cuadrado en esquina (resultado binario, sin fórmula)
      - Sin porcentaje sobre cada barra (evita sobrecarga visual)
    """
    from canal.modulador import modular, simbolos_a_etiquetas
    from scipy.stats     import chisquare

    etiq     = simbolos_a_etiquetas(modular(bits, A=A), A=A)
    counts   = np.bincount(etiq, minlength=16)
    freqs    = counts / counts.sum()
    freq_uni = 1.0 / 16
    chi2, pval = chisquare(counts)

    # Colores: azul neutro por defecto; rojo solo si desvío > 10 %
    desvios = (freqs - freq_uni) / freq_uni * 100
    COLOR_BASE  = '#4C72B0'
    COLOR_DESV  = '#C0392B'
    colores = [COLOR_DESV if abs(d) > 10 else COLOR_BASE for d in desvios]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(range(16), freqs * 100,
           color=colores, edgecolor='white', linewidth=0.6,
           alpha=0.85, zorder=3)

    # Banda de tolerancia ±5 % (zona aceptable alrededor de equiprobable)
    ax.axhspan(freq_uni * 100 * 0.95, freq_uni * 100 * 1.05,
               color='#888888', alpha=0.10, zorder=1,
               label='Tolerancia ±5 %')

    # Línea de referencia equiprobable — gris, discreta
    ax.axhline(freq_uni * 100, color='#888888', linewidth=1.2,
               linestyle='--', zorder=4,
               label=f'Equiprobable = {freq_uni*100:.2f} %')

    # Resultado del test chi-cuadrado — solo conclusión, sin fórmula
    es_uniforme = pval >= 0.05
    color_p  = '#27AE60' if es_uniforme else '#C0392B'
    texto_p  = 'Distribución uniforme' if es_uniforme else 'Distribución no uniforme'
    ax.text(0.98, 0.97,
            f'{texto_p}\n$p = {pval:.3f}$ (α = 0.05)',
            transform=ax.transAxes, fontsize=8.5,
            va='top', ha='right', color=color_p,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color_p, alpha=0.9))

    # Ejes
    ax.set_xlabel('Símbolo 16-QAM', fontsize=11)
    ax.set_ylabel('Frecuencia relativa (%)', fontsize=11)
    ax.set_xticks(range(16))
    ax.set_xticklabels([str(i) for i in range(16)], fontsize=9)
    ax.set_ylim(0, max(freqs * 100) * 1.22)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.set_title(f'{titulo}\n$N = {n_sim:,}$ símbolos',
                 fontsize=11, pad=10)

    plt.tight_layout()
    if guardar:
        _guardar(fig, nombre_archivo, dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 8. Histograma para artículo — PRE scrambling
# ---------------------------------------------------------------------------
def histograma_articulo_pre(fuente_dict, A=1.0, n_simbolos=50_000,
                            guardar=False, dir_resultados="resultados/"):
    """
    Histograma de frecuencia de símbolos 16-QAM ANTES del scrambling.
    Versión minimalista para publicación en artículo científico.

    Parámetros
    ----------
    fuente_dict   : dict devuelto por la función de fuente
    A             : amplitud de la constelación
    n_simbolos    : cantidad de símbolos a analizar
    guardar       : si True, guarda la figura en dir_resultados
    dir_resultados: carpeta de salida
    """
    from fuente.fuente import extraer_segmento

    n_sim = min(n_simbolos, fuente_dict['n_simbolos'])
    bits  = extraer_segmento(fuente_dict, 0, n_sim, usar_scrambling=False)

    _histograma_articulo(
        bits       = bits,
        n_sim      = n_sim,
        A          = A,
        titulo     = 'Distribución de símbolos — sin scrambling',
        nombre_archivo = 'histograma_articulo_pre_scrambling.png',
        guardar        = guardar,
        dir_resultados = dir_resultados,
    )


# ---------------------------------------------------------------------------
# 9. Histograma para artículo — POST scrambling
# ---------------------------------------------------------------------------
def histograma_articulo_post(fuente_dict, A=1.0, n_simbolos=50_000,
                             scrambling_seed=42, scrambling_poly=0x400007,
                             guardar=False, dir_resultados="resultados/"):
    """
    Histograma de frecuencia de símbolos 16-QAM DESPUÉS del scrambling.
    Versión minimalista para publicación en artículo científico.

    Parámetros
    ----------
    fuente_dict     : dict devuelto por la función de fuente
    A               : amplitud de la constelación
    n_simbolos      : cantidad de símbolos a analizar
    scrambling_seed : semilla del scrambler LFSR
    scrambling_poly : polinomio generador del scrambler
    guardar         : si True, guarda la figura en dir_resultados
    dir_resultados  : carpeta de salida
    """
    from fuente.fuente import extraer_segmento

    n_sim = min(n_simbolos, fuente_dict['n_simbolos'])
    bits  = extraer_segmento(fuente_dict, 0, n_sim,
                             usar_scrambling=True,
                             scrambling_seed=scrambling_seed,
                             scrambling_poly=scrambling_poly)

    _histograma_articulo(
        bits       = bits,
        n_sim      = n_sim,
        A          = A,
        titulo     = 'Distribución de símbolos — con scrambling',
        nombre_archivo = 'histograma_articulo_post_scrambling.png',
        guardar        = guardar,
        dir_resultados = dir_resultados,
    )
