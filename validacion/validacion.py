"""
validacion/validacion.py
========================
Fase 0 del benchmark: validación del simulador.

Objetivo:
  Demostrar que la cadena modulador 16-QAM → canal AWGN → demodulador Bayesiano
  reproduce fielmente la curva teórica de BER. Esta validación es el prerrequisito
  científico que justifica el uso del simulador como entorno de entrenamiento para ML.

Criterio estadístico:
  La BER simulada solo es confiable si se cuentan al menos N_ERRORES_MIN errores.
  La regla práctica en telecomunicaciones es N_errores >= 100.
  Para garantizarlo, el número de símbolos por punto se calcula adaptativamente:

      N_simb = ceil(N_ERRORES_MIN / (BER_teorica * BITS_POR_SIMBOLO))

  Si N_simb excede N_MAX_SIMBOLOS, el punto se simula con N_MAX y se marca
  en la gráfica como estadísticamente no confiable.

  Esto explica por qué la curva simulada puede aparecer por debajo de la teórica
  a alto Eb/N0: no es un logro físico sino una fluctuación estadística por
  muestra insuficiente. El detector Bayesiano es óptimo y su BER real nunca
  puede ser inferior a la teórica.
"""

import numpy as np
import matplotlib.pyplot as plt

from canal.canal_awgn  import ber_teorica_16qam, calcular_sigma, agregar_ruido
from canal.modulador   import modular, demodular_bayes_vectorizado
from fuente.fuente     import extraer_segmento


# ---------------------------------------------------------------------------
# Parámetros de la validación
# ---------------------------------------------------------------------------
N_ERRORES_MIN    = 100        # Mínimo de errores para estimación confiable
N_MAX_SIMBOLOS   = 2_000_000  # Límite práctico por punto (tiempo de cómputo)
A_VALIDACION     = 1.0
BITS_POR_SIMBOLO = 4          # 16-QAM


# ---------------------------------------------------------------------------
# Función principal de validación
# ---------------------------------------------------------------------------

def validar_simulador(
    fuente_dict:    dict,
    Eb_N0_range_dB: list,
    seed:           int  = 42,
    guardar_figura: bool = False,
    dir_resultados: str  = "resultados/"
):
    print("\n" + "="*60)
    print("  FASE 0 — VALIDACIÓN DEL SIMULADOR")
    print("="*60)
    print(f"  Criterio: mínimo {N_ERRORES_MIN} errores por punto")
    print(f"  Límite práctico: {N_MAX_SIMBOLOS:,} símbolos por punto")
    print(f"  Barrido: {Eb_N0_range_dB} dB\n")

    rng          = np.random.default_rng(seed)
    ber_simulada = np.zeros(len(Eb_N0_range_dB))
    n_errores    = np.zeros(len(Eb_N0_range_dB), dtype=int)
    confiable    = np.ones(len(Eb_N0_range_dB),  dtype=bool)

    n_disponibles = fuente_dict['n_simbolos']

    for i, Eb_N0_dB in enumerate(Eb_N0_range_dB):

        # N adaptativo para este punto
        ber_teo = ber_teorica_16qam([Eb_N0_dB])[0]
        n_simb_necesarios = int(np.ceil(N_ERRORES_MIN / (ber_teo * BITS_POR_SIMBOLO)))
        n_simb = min(n_simb_necesarios, N_MAX_SIMBOLOS, n_disponibles)
        es_confiable = (n_simb >= n_simb_necesarios)

        # Transmisión y demodulación
        bits_val    = extraer_segmento(fuente_dict, inicio_simbolo=0, n_simbolos=n_simb)
        simbolos_tx = modular(bits_val, A=A_VALIDACION)
        sigma       = calcular_sigma(Eb_N0_dB, A_VALIDACION)
        simbolos_rx = agregar_ruido(simbolos_tx, sigma, rng)
        bits_det    = demodular_bayes_vectorizado(simbolos_rx, A=A_VALIDACION)

        errores = int(np.sum(bits_val != bits_det))
        ber_sim = errores / len(bits_val) if errores > 0 else 0.0

        ber_simulada[i] = ber_sim
        n_errores[i]    = errores
        confiable[i]    = es_confiable

        estado = "OK" if es_confiable else "INSUFICIENTE"
        print(f"  Eb/N0 = {Eb_N0_dB:4.1f} dB  |  N = {n_simb:>9,}  |  "
              f"errores = {errores:>6}  |  "
              f"BER_sim = {ber_sim:.2e}  |  BER_teo = {ber_teo:.2e}  |  {estado}")

    no_confiables = [Eb_N0_range_dB[i] for i in range(len(Eb_N0_range_dB))
                     if not confiable[i]]
    if no_confiables:
        print(f"\n  Puntos estadisticamente no confiables: {no_confiables} dB")
        print(f"  (menos de {N_ERRORES_MIN} errores contados)")
        print(f"  Aumentar N_MAX_SIMBOLOS en validacion.py para mejorarlos.")

    aprobado = _graficar_validacion(
        Eb_N0_range_dB, ber_simulada, confiable, n_errores,
        guardar_figura, dir_resultados
    )

    return ber_simulada, aprobado


# ---------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------

def _graficar_validacion(Eb_N0_range_dB, ber_simulada, confiable, n_errores,
                         guardar_figura, dir_resultados):

    Eb_N0_fino = np.linspace(min(Eb_N0_range_dB) - 1, max(Eb_N0_range_dB) + 1, 400)
    ber_teo    = ber_teorica_16qam(Eb_N0_fino)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Curva teórica
    ax.semilogy(Eb_N0_fino, ber_teo,
                color='steelblue', linewidth=2.0, label='BER teórica 16-QAM', zorder=5)

    # Puntos confiables
    xc = [Eb_N0_range_dB[i] for i in range(len(Eb_N0_range_dB)) if confiable[i]]
    yc = [ber_simulada[i]   for i in range(len(Eb_N0_range_dB)) if confiable[i]]
    if xc:
        ax.semilogy(xc, yc, 'o-', color='tomato', linewidth=1.5, markersize=8,
                    label='BER simulada (Bayes) — confiable', zorder=6)

    # Puntos no confiables
    xn = [Eb_N0_range_dB[i] for i in range(len(Eb_N0_range_dB))
          if not confiable[i] and ber_simulada[i] > 0]
    yn = [ber_simulada[i]   for i in range(len(Eb_N0_range_dB))
          if not confiable[i] and ber_simulada[i] > 0]
    if xn:
        ax.semilogy(xn, yn, 's--', color='darkorange', linewidth=1.0, markersize=8,
                    markerfacecolor='none', markeredgewidth=1.5,
                    label=f'BER simulada — estadistica insuficiente (<{N_ERRORES_MIN} errores)',
                    zorder=6)

    # Anotaciones
    ber_teo_puntos = ber_teorica_16qam(Eb_N0_range_dB)
    for i, (x, y_sim, y_teo) in enumerate(zip(Eb_N0_range_dB, ber_simulada, ber_teo_puntos)):
        if y_sim <= 0:
            continue
        if confiable[i]:
            desvio = abs(y_sim - y_teo) / y_teo * 100
            ax.annotate(f'{desvio:.1f}%\n({n_errores[i]} err)',
                        xy=(x, y_sim), xytext=(5, 4), textcoords='offset points',
                        fontsize=7.5, color='dimgray')
        else:
            ax.annotate(f'({n_errores[i]} err)',
                        xy=(x, y_sim), xytext=(5, -14), textcoords='offset points',
                        fontsize=7.5, color='darkorange')

    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=13)
    ax.set_ylabel(r'Probabilidad de error de bit $P_b$', fontsize=13)
    ax.set_title('Validacion del simulador\nBER Bayes simulada vs. curva teorica 16-QAM',
                 fontsize=13)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, loc='lower left')

    fig.text(0.5, -0.03,
             f'Nota: puntos confiables requieren >= {N_ERRORES_MIN} errores contados. '
             'El detector Bayesiano es optimo — su BER real nunca puede ser inferior a la teorica.\n'
             'Desviaciones hacia abajo en puntos de alto Eb/N0 son artefactos estadisticos '
             'por muestra insuficiente, no logros fisicos.',
             ha='center', fontsize=8.5, color='dimgray', style='italic')

    plt.tight_layout()

    if guardar_figura:
        import os
        os.makedirs(dir_resultados, exist_ok=True)
        ruta = f"{dir_resultados}validacion_simulador.png"
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        print(f"\n  [Validacion] Figura guardada en '{ruta}'")

    plt.show(block=False)
    plt.pause(0.5)

    print("\n" + "-"*60)
    print("  La grafica de validacion se ha mostrado.")
    print("  Puntos solidos (rojo)   : estadisticamente confiables.")
    print("  Puntos vacios (naranja) : estadistica insuficiente, ignorar.")
    print("-"*60)

    while True:
        respuesta = input("\n  El simulador esta validado? [s/n]: ").strip().lower()
        if respuesta in ('s', 'si', 'y', 'yes'):
            print("  Simulador validado. Continuando con el benchmark ML...\n")
            plt.close(fig)
            return True
        elif respuesta in ('n', 'no'):
            print("  Simulador NO aprobado. Revisar parametros en config.py")
            plt.close(fig)
            return False
        else:
            print("  Por favor ingrese 's' o 'n'.")


# ---------------------------------------------------------------------------
# Utilidad: tabla de N necesario por punto
# ---------------------------------------------------------------------------

def calcular_n_por_punto(Eb_N0_range_dB: list) -> None:
    """Imprime la tabla de símbolos necesarios por punto para N_ERRORES_MIN errores."""
    print(f"\n  Simbolos necesarios para {N_ERRORES_MIN} errores minimos por punto:\n")
    print(f"  {'Eb/N0':>8} {'BER teorica':>14} {'N simbolos':>14} {'Confiable con N_MAX':>22}")
    print("  " + "-"*62)
    for EbN0_dB in Eb_N0_range_dB:
        ber = ber_teorica_16qam([EbN0_dB])[0]
        n_simb = int(np.ceil(N_ERRORES_MIN / (ber * BITS_POR_SIMBOLO)))
        ok = "OK" if n_simb <= N_MAX_SIMBOLOS else f"NO (necesita {n_simb:,})"
        print(f"  {EbN0_dB:>6} dB  {ber:>14.3e}  {n_simb:>14,}  {ok:>22}")
