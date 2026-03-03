"""
graficas/graficas.py
====================
Visualizaciones del benchmark unificado.

Figuras generadas:
  1. curva_ber_principal()    — BER vs Eb/N0 (escala log completa)
  2. curva_ber_zoom()         — BER vs Eb/N0 (zoom región útil)
  3. grafica_degradacion()    — degradación en dB respecto a Bayes (NUEVA)
  4. tabla_metricas()         — tabla visual de métricas comparativas
  5. regiones_decision()      — regiones de decisión en plano IQ
  6. historial_loss()         — curvas de pérdida de redes neuronales
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
    'Teorica'       : '#000000',
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
    ax.semilogy(Eb_N0_fino, ber_teo_fino,
                color=_COLORES['Teorica'], linewidth=2.5,
                linestyle='-', label='BER Teorica 16-QAM', zorder=10)

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
             'Marcador vacio (···): estadistica insuficiente (<100 errores).',
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
    ax.set_title('BER vs $E_b/N_0$ — Zoom region de interes\n'
                 'Canal AWGN — Protocolo de entrenamiento realista', fontsize=12)
    fig.text(0.5, -0.02,
             'Marcador vacio (···): estadistica insuficiente (<100 errores).',
             ha='center', fontsize=8.5, color='dimgray', style='italic')
    plt.tight_layout()
    if guardar:
        _guardar(fig, 'ber_curva_zoom.png', dir_resultados)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Gráfica de degradación (NUEVA)
# ---------------------------------------------------------------------------
def grafica_degradacion(registros, guardar=False, dir_resultados="resultados/"):
    """
    Degradación en dB respecto al detector Bayesiano.

    Para cada clasificador ML y cada nivel de BER, calcula cuántos dB
    adicionales necesita respecto a Bayes para alcanzar esa misma BER.

    Degradacion(BER) = Eb/N0_clasificador(BER) - Eb/N0_Bayes(BER)

    Un valor de 0 dB significa rendimiento igual al Bayesiano.
    Un valor positivo indica cuántos dB extra se necesitan (peor rendimiento).

    Esta representación es especialmente útil cuando las curvas BER son
    visualmente muy similares: pequeñas diferencias en dB son claramente
    visibles aquí aunque no lo sean en la escala logarítmica de BER.
    """
    # Curva teórica densa para interpolar Eb/N0 a partir de BER objetivo
    Eb_N0_denso = np.linspace(-2, 20, 5000)
    ber_teo_den = ber_teorica_16qam(Eb_N0_denso)

    # Identificar el registro Bayes (referencia)
    reg_bayes = next((r for r in registros
                      if 'bayes' in r['nombre'].lower()), None)
    if reg_bayes is None:
        print("  [Graficas] No se encontró el clasificador Bayes. "
              "Gráfica de degradación omitida.")
        return

    # Rango de BER para el que tiene sentido calcular degradación
    # (solo puntos confiables de Bayes)
    x_b = np.array(reg_bayes['Eb_N0_dB'])
    y_b = np.array(reg_bayes['ber'], dtype=float)
    c_b = np.array(reg_bayes.get('confiable', [True]*len(x_b)), dtype=bool)

    ber_validas = y_b[c_b & (y_b > 0)]
    if len(ber_validas) < 2:
        print("  [Graficas] Datos insuficientes para gráfica de degradación.")
        return

    # BER targets: puntos confiables de Bayes, en escala log uniforme
    ber_min = max(ber_validas.min(), 1e-6)
    ber_max = ber_validas.max()
    ber_targets = np.logspace(np.log10(ber_min), np.log10(ber_max), 200)

    # Función que dado una BER devuelve el Eb/N0 teórico necesario
    # (interpolación inversa sobre la curva teórica densa)
    try:
        interp_teo = interp1d(ber_teo_den[::-1], Eb_N0_denso[::-1],
                              kind='linear', bounds_error=False,
                              fill_value=(Eb_N0_denso[-1], Eb_N0_denso[0]))
        EbN0_bayes_teo = interp_teo(ber_targets)
    except Exception:
        print("  [Graficas] Error en interpolacion para degradacion.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    # Línea de referencia en 0 dB (Bayes teórico)
    ax.axhline(0, color=_COLORES['Teorica'], linewidth=2.0,
               linestyle='-', label='Bayes teorico (referencia 0 dB)', zorder=10)

    clf_sin_bayes = [r for r in registros if 'bayes' not in r['nombre'].lower()]

    for reg in clf_sin_bayes:
        x   = np.array(reg['Eb_N0_dB'])
        y   = np.array(reg['ber'], dtype=float)
        c   = np.array(reg.get('confiable', [True]*len(x)), dtype=bool)
        color  = _color(reg['nombre'])
        marker = _marker(reg['nombre'])

        # Solo usar puntos confiables con BER > 0
        mask = c & (y > 0)
        if mask.sum() < 2:
            continue

        x_ok = x[mask]
        y_ok = y[mask]

        # Para cada BER target, interpolar Eb/N0 del clasificador
        try:
            interp_clf = interp1d(y_ok[::-1], x_ok[::-1],
                                  kind='linear', bounds_error=False,
                                  fill_value=np.nan)
            EbN0_clf = interp_clf(ber_targets)
        except Exception:
            continue

        degradacion = EbN0_clf - EbN0_bayes_teo

        # Filtrar NaN y valores fuera del rango de datos
        mask_plot = np.isfinite(degradacion) & \
                    (ber_targets >= y_ok.min() * 0.9) & \
                    (ber_targets <= y_ok.max() * 1.1)

        if mask_plot.sum() < 2:
            continue

        ax.plot(ber_targets[mask_plot], degradacion[mask_plot],
                color=color, linewidth=1.8, label=reg['nombre'])

        # Marcar el punto de máxima degradación
        idx_max = np.argmax(np.abs(degradacion[mask_plot]))
        bx = ber_targets[mask_plot][idx_max]
        dy = degradacion[mask_plot][idx_max]
        ax.plot(bx, dy, marker=marker, color=color, markersize=8, zorder=5)

    ax.set_xscale('log')
    ax.invert_xaxis()   # BER decrece de izquierda a derecha (SNR crece)
    ax.set_xlabel(r'BER objetivo ($P_b$)', fontsize=12)
    ax.set_ylabel(r'Degradacion respecto a Bayes (dB)', fontsize=12)
    ax.set_title('Degradacion en $E_b/N_0$ respecto al detector Bayesiano\n'
                 'Positivo = mas dB necesarios para la misma BER', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3,
               alpha=0.04, color='red', label='_nolegend_')

    # Nota explicativa
    fig.text(0.5, -0.02,
             'Degradacion = Eb/N0 necesario por el clasificador ML menos Eb/N0 del Bayes teorico '
             'para la misma BER.\n'
             '0 dB = rendimiento identico al optimo. '
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
                'T. entrenamiento\n(s, promedio)', 'T. inferencia\n(us/simbolo)']
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

    plt.suptitle(f'Regiones de decision — $E_b/N_0$ = {Eb_N0_dB} dB',
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
    ax.set_xlabel('Epoca', fontsize=12)
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
# 7. Histograma de símbolos transmitidos
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

    Mostrar ambas permite argumentar en el artículo:
      - La fuente de voz + Ley µ produce bits no equiprobables (Figura A)
      - El scrambling corrige esto, validando el supuesto del Bayesiano (Figura B)
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

    def _graficar(bits, titulo, nombre_archivo):
        freqs, desvios, counts, p1_bits, chi2, pval = _analizar(bits)
        freq_uni = 1.0 / 16
        iq_todos = A * _TODOS_IQ
        p1_global = float(bits.mean())

        fig = plt.figure(figsize=(16, 5))
        gs  = fig.add_gridspec(1, 3, wspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # (a) Histograma de frecuencias de símbolos
        colores_hist = ['#E53935' if abs(d) > 10 else
                        '#FB8C00' if abs(d) > 5  else
                        '#43A047' for d in desvios]
        ax1.bar(range(16), freqs * 100, color=colores_hist,
                edgecolor='white', linewidth=0.5)
        ax1.axhline(freq_uni * 100, color='black', linewidth=1.5,
                    linestyle='--', label=f'Equiprobable ({freq_uni*100:.2f}%)')
        for i, (f, d) in enumerate(zip(freqs, desvios)):
            if abs(d) > 5:
                ax1.annotate(f'{d:+.1f}%', xy=(i, f*100 + 0.1),
                             ha='center', fontsize=7, color='dimgray')
        ax1.set_xlabel('Símbolo (índice 0–15)', fontsize=11)
        ax1.set_ylabel('Frecuencia relativa (%)', fontsize=11)
        ax1.set_title('(a) Distribución de símbolos', fontsize=11)
        ax1.set_xticks(range(16))
        ax1.legend(fontsize=9)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        color_stat = '#E53935' if pval < 0.05 else '#43A047'
        ax1.text(0.02, 0.97,
                 f'χ²={chi2:.1f}  p={pval:.4f}\n'
                 f'{"No equiprobable" if pval < 0.05 else "Equiprobable"} (α=0.05)',
                 transform=ax1.transAxes, fontsize=8.5, va='top', color=color_stat,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # (b) Constelación coloreada por frecuencia
        sc = ax2.scatter(iq_todos[:, 0], iq_todos[:, 1],
                         c=freqs * 100, cmap='RdYlGn_r', s=400,
                         edgecolors='black', linewidths=0.8,
                         vmin=(freq_uni*100)*0.7, vmax=(freq_uni*100)*1.3, zorder=5)
        cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label('Frecuencia (%)', fontsize=9)
        cb.ax.axhline(freq_uni * 100, color='black', linewidth=1.5, linestyle='--')
        for iq, freq in zip(iq_todos, freqs):
            ax2.annotate(f'{freq*100:.1f}%', xy=iq, xytext=(0, 12),
                         textcoords='offset points', ha='center', fontsize=7.5)
        ax2.set_xlabel('Componente I', fontsize=11)
        ax2.set_ylabel('Componente Q', fontsize=11)
        ax2.set_title('(b) Constelación: frecuencia por símbolo\n'
                      'Verde=equiprobable  Rojo=desviado', fontsize=10)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.axvline(0, color='gray', linewidth=0.5)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_aspect('equal')

        # (c) P(1) por posición de bit
        posiciones   = ['b₀ (MSB)', 'b₁', 'b₂', 'b₃ (LSB)']
        colores_bits = ['#43A047' if abs(p - 0.5) < 0.05 else '#E53935'
                        for p in p1_bits]
        ax3.bar(posiciones, p1_bits * 100, color=colores_bits,
                edgecolor='white', linewidth=0.5)
        ax3.axhline(50, color='black', linewidth=1.5, linestyle='--',
                    label='Equiprobable (50%)')
        ax3.set_ylim(0, 100)
        ax3.set_xlabel('Posición de bit en símbolo', fontsize=11)
        ax3.set_ylabel('P(bit=1) (%)', fontsize=11)
        ax3.set_title('(c) Equiprobabilidad por posición de bit\n'
                      'Verde: |P(1)−50%| < 5pp   Rojo: desvío > 5pp', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
        for i, p in enumerate(p1_bits):
            ax3.annotate(f'{p*100:.1f}%\n({(p-0.5)*100:+.1f}pp)',
                         xy=(i, p*100 + 1), ha='center', fontsize=9)

        fig.suptitle(
            f'{titulo}\n'
            f'P(1) global = {p1_global*100:.2f}%  |  '
            f'Desvío = {abs(p1_global-0.5)*100:.2f} pp  |  '
            f'N = {n_sim:,} símbolos',
            fontsize=12, fontweight='bold')
        fig.text(0.5, -0.03,
                 "El detector Bayesiano (ML) es óptimo bajo el supuesto de símbolos "
                 "equiprobables. Un desvío significativo implica que el detector MAP "
                 "con probabilidades a priori reales sería más apropiado.",
                 ha='center', fontsize=8.5, color='dimgray', style='italic')
        plt.tight_layout()
        if guardar:
            _guardar(fig, nombre_archivo, dir_resultados)
        plt.show()

    # Figura A: fuente cruda — SIEMPRE se muestra
    bits_crudo = extraer_segmento(fuente_dict, 0, n_sim, usar_scrambling=False)
    _graficar(bits_crudo,
              'Distribución de símbolos — ANTES del scrambling\n'
              'Fuente: voz + Ley µ (distribución real de la fuente)',
              'histograma_sin_scrambling.png')

    # Figura B: post-scrambling — solo si está activo
    if usar_scrambling:
        bits_scram = extraer_segmento(fuente_dict, 0, n_sim,
                                      usar_scrambling=True,
                                      scrambling_seed=scrambling_seed,
                                      scrambling_poly=scrambling_poly)
        _graficar(bits_scram,
                  'Distribución de símbolos — DESPUÉS del scrambling\n'
                  'Condición usada en el benchmark (equiprobabilidad garantizada)',
                  'histograma_con_scrambling.png')
    else:
        print("\n  [Histograma] Solo se muestra la distribución de la fuente cruda.")
        print("  [Histograma] Activar USAR_SCRAMBLING=True en config.py para ver "
              "la comparación completa antes/después.")


