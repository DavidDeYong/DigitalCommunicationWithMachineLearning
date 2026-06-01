"""
metricas/metricas.py
====================
Cálculo y almacenamiento de todas las métricas del benchmark.

Métricas por clasificador por punto Eb/N0:
  - BER  (Bit Error Rate)
  - Accuracy de símbolo
  - Tiempo de entrenamiento (segundos)
  - Tiempo de inferencia por símbolo (microsegundos)
  - Mejores hiperparámetros encontrados
  - n_errores        : errores contados en el test
  - n_test_simbolos  : símbolos usados en el test
  - confiable        : True si n_errores >= N_ERRORES_MIN
  - n_necesario      : símbolos que se necesitarían para ser confiable
"""

import numpy as np
from canal.modulador    import etiquetas_a_bits
from canal.canal_awgn  import ber_teorica_16qam

# Criterio estadístico (coherente con validacion.py)
N_ERRORES_MIN    = 100
BITS_POR_SIMBOLO = 4


# ---------------------------------------------------------------------------
# Confiabilidad estadística
# ---------------------------------------------------------------------------

def evaluar_confiabilidad(n_errores: int, Eb_N0_dB: float, n_test_simbolos: int) -> dict:
    """
    Determina si un punto es estadísticamente confiable y calcula cuántos
    símbolos serían necesarios para serlo.

    Returns
    -------
    dict con:
        confiable       : bool
        n_errores       : int
        n_test_simbolos : int
        n_necesario     : int   símbolos requeridos para N_ERRORES_MIN errores
    """
    # Usamos la BER teórica como referencia para calcular N necesario,
    # ya que la BER real del clasificador es desconocida a priori.
    ber_ref   = ber_teorica_16qam([Eb_N0_dB])[0]
    n_nec     = int(np.ceil(N_ERRORES_MIN / (ber_ref * BITS_POR_SIMBOLO)))
    confiable = (n_errores >= N_ERRORES_MIN)

    return {
        'confiable'       : confiable,
        'n_errores'       : n_errores,
        'n_test_simbolos' : n_test_simbolos,
        'n_necesario'     : n_nec,
    }


# ---------------------------------------------------------------------------
# Cálculo de BER y accuracy
# ---------------------------------------------------------------------------

def calcular_ber(bits_referencia: np.ndarray, etiquetas_det: np.ndarray) -> tuple:
    """
    Calcula la BER y cuenta los errores.

    Returns
    -------
    ber      : float
    errores  : int
    """
    bits_det = etiquetas_a_bits(etiquetas_det)
    n        = min(len(bits_referencia), len(bits_det))
    errores  = int(np.sum(bits_referencia[:n] != bits_det[:n]))
    ber      = float(errores) / n
    return ber, errores


def calcular_accuracy(etiquetas_ref: np.ndarray, etiquetas_det: np.ndarray) -> float:
    n = min(len(etiquetas_ref), len(etiquetas_det))
    return float(np.sum(etiquetas_ref[:n] == etiquetas_det[:n])) / n


# ---------------------------------------------------------------------------
# Estructura de resultados
# ---------------------------------------------------------------------------

def nuevo_registro_clasificador(nombre: str) -> dict:
    return {
        'nombre'            : nombre,
        'Eb_N0_dB'          : [],
        'ber'               : [],
        'accuracy'          : [],
        't_entrenamiento_s' : [],
        't_inferencia_us'   : [],
        'flops_inferencia'  : [],   # FLOPs por símbolo (hardware-independiente)
        'mejores_hiperparams': [],
        # Información estadística por punto
        'n_errores'         : [],
        'n_test_simbolos'   : [],
        'confiable'         : [],
        'n_necesario'       : [],
    }


def registrar_punto(
    registro:         dict,
    Eb_N0_dB:         float,
    bits_ref:         np.ndarray,
    etiquetas_ref:    np.ndarray,
    etiquetas_det:    np.ndarray,
    t_entrenamiento:  float,
    t_inferencia_1:   float,
    mejores_params:   dict,
    n_test_simbolos:  int = None,
    flops_inferencia: int = 0,
) -> None:
    ber, errores = calcular_ber(bits_ref, etiquetas_det)
    accuracy     = calcular_accuracy(etiquetas_ref, etiquetas_det)

    n_simb = n_test_simbolos if n_test_simbolos is not None else len(etiquetas_det)
    info_conf = evaluar_confiabilidad(errores, Eb_N0_dB, n_simb)

    registro['Eb_N0_dB'].append(Eb_N0_dB)
    registro['ber'].append(ber)
    registro['accuracy'].append(accuracy)
    registro['t_entrenamiento_s'].append(t_entrenamiento)
    registro['t_inferencia_us'].append(t_inferencia_1 * 1e6)
    registro['flops_inferencia'].append(flops_inferencia)
    registro['mejores_hiperparams'].append(mejores_params)
    registro['n_errores'].append(errores)
    registro['n_test_simbolos'].append(n_simb)
    registro['confiable'].append(info_conf['confiable'])
    registro['n_necesario'].append(info_conf['n_necesario'])


# ---------------------------------------------------------------------------
# Resúmenes en consola
# ---------------------------------------------------------------------------

def imprimir_resumen(registro: dict) -> None:
    nombre = registro['nombre']
    print(f"\n  ── {nombre} ──")
    print(f"  {'Eb/N0':>6} {'BER':>12} {'Acc':>8} "
          f"{'t_train':>9} {'t_inf(µs)':>10} {'errores':>8} {'confiable':>10}")
    print("  " + "-"*72)
    for i in range(len(registro['Eb_N0_dB'])):
        ber     = registro['ber'][i]
        ber_str = f"{ber:.2e}" if ber > 0 else "  0 (s/err)"
        conf    = "OK" if registro['confiable'][i] else \
                  f"NO ({registro['n_errores'][i]} err)"
        print(
            f"  {registro['Eb_N0_dB'][i]:>6.1f} "
            f"{ber_str:>12} "
            f"{registro['accuracy'][i]:>8.4f} "
            f"{registro['t_entrenamiento_s'][i]:>9.2f} "
            f"{registro['t_inferencia_us'][i]:>10.4f} "
            f"{registro['n_errores'][i]:>8} "
            f"{conf:>10}"
        )


def imprimir_advertencias_estadisticas(registros: list) -> None:
    """
    Imprime una tabla consolidada de todos los puntos no confiables
    y cuántos símbolos se necesitarían para hacerlos confiables.
    Pensada para incluir en el artículo.
    """
    hay_problemas = False
    lineas = []

    for reg in registros:
        for i in range(len(reg['Eb_N0_dB'])):
            if not reg['confiable'][i]:
                hay_problemas = True
                lineas.append({
                    'clf'       : reg['nombre'],
                    'Eb_N0'     : reg['Eb_N0_dB'][i],
                    'errores'   : reg['n_errores'][i],
                    'n_usado'   : reg['n_test_simbolos'][i],
                    'n_nec'     : reg['n_necesario'][i],
                })

    if not hay_problemas:
        print("\n  Todos los puntos del barrido son estadisticamente confiables.")
        return

    print(f"\n  Puntos con estadistica insuficiente (< {N_ERRORES_MIN} errores):")
    print(f"\n  {'Clasificador':<20} {'Eb/N0':>6} {'Errores':>8} "
          f"{'N usado':>12} {'N necesario':>14}  {'Factor':>8}")
    print("  " + "-"*78)
    for l in lineas:
        factor = l['n_nec'] / l['n_usado'] if l['n_usado'] > 0 else float('inf')
        print(f"  {l['clf']:<20} {l['Eb_N0']:>6.1f} {l['errores']:>8} "
              f"{l['n_usado']:>12,} {l['n_nec']:>14,}  x{factor:>6.0f}")

    print(f"\n  'Factor' indica cuantas veces mas simbolos se necesitarian.")
    print(f"  Para el articulo, estos puntos se grafican con marcador especial.")


def construir_tabla_comparativa(registros: list) -> dict:
    tabla = {}
    for reg in registros:
        flops = reg.get('flops_inferencia', [])
        tabla[reg['nombre']] = {
            'ber_media'            : np.mean(reg['ber']),
            'accuracy_media'       : np.mean(reg['accuracy']),
            't_train_media_s'      : np.mean(reg['t_entrenamiento_s']),
            't_inferencia_media_us': np.mean(reg['t_inferencia_us']),
            'flops_inferencia'     : int(flops[0]) if flops else 0,
        }
    return tabla
