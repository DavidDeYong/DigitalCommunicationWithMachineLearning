"""
canal/modulador.py
==================
Modulador y demodulador 16-QAM con mapeo de Gray.

Convención de constelación:
  - 16 puntos en el plano I/Q con valores en {-3, -1, +1, +3}
  - Energía promedio por símbolo: Es = (1/16) * Σ(I²+Q²) = 10  [con A=1]
  - Energía promedio por bit:     Eb = Es / 4 = 2.5             [con A=1]
  - La amplitud A escala la constelación para ajustar Eb al Eb/N0 deseado.

Mapeo Gray (4 bits → símbolo IQ):
  Los bits se dividen en dos grupos de 2 (MSB→eje I, LSB→eje Q)
  con código Gray para minimizar BER en errores de símbolo vecino.
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Constelación 16-QAM con mapeo Gray
# ---------------------------------------------------------------------------
# Mapeo Gray para 2 bits → valor en {-3,-1,+1,+3}
# 00 → -3,  01 → -1,  11 → +1,  10 → +3
_GRAY_2BIT = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): +1,
    (1, 0): +3,
}
_GRAY_2BIT_INV = {v: k for k, v in _GRAY_2BIT.items()}

# Construir tabla completa: 4 bits → (I, Q)  y su inversa (I, Q) → 4 bits
_MAPA_BITS_A_IQ: dict[tuple, tuple] = {}
_MAPA_IQ_A_BITS: dict[tuple, tuple] = {}

for b0 in (0, 1):
    for b1 in (0, 1):
        for b2 in (0, 1):
            for b3 in (0, 1):
                I = _GRAY_2BIT[(b0, b1)]
                Q = _GRAY_2BIT[(b2, b3)]
                _MAPA_BITS_A_IQ[(b0, b1, b2, b3)] = (I, Q)
                _MAPA_IQ_A_BITS[(I, Q)] = (b0, b1, b2, b3)

# Listas numpy para operaciones vectorizadas
_TODOS_IQ   = np.array(list(_MAPA_IQ_A_BITS.keys()),   dtype=np.float64)   # shape (16, 2)
_TODOS_BITS = np.array(list(_MAPA_IQ_A_BITS.values()), dtype=np.int8)      # shape (16, 4)

# Energía promedio por símbolo con A=1
ENERGIA_SIMBOLO_NORM = np.mean(np.sum(_TODOS_IQ ** 2, axis=1))  # = 10.0
ENERGIA_BIT_NORM     = ENERGIA_SIMBOLO_NORM / 4                  # = 2.5


# ---------------------------------------------------------------------------
# Funciones de modulación
# ---------------------------------------------------------------------------

def bits_a_simbolos(bits: np.ndarray) -> np.ndarray:
    """
    Convierte un bitstream (múltiplo de 4) a array de índices de símbolo (0-15).

    Parameters
    ----------
    bits : np.ndarray int8, longitud múltiplo de 4

    Returns
    -------
    indices : np.ndarray int, shape (N_simbolos,)
    """
    n = len(bits)
    if n % 4 != 0:
        raise ValueError(f"[Modulador] El bitstream debe tener longitud múltiplo de 4, tiene {n}.")

    grupos = bits.reshape(-1, 4)
    indices = []
    for grupo in grupos:
        clave = tuple(int(b) for b in grupo)
        iq    = _MAPA_BITS_A_IQ[clave]
        # Obtener índice en la lista de símbolos
        idx = np.where(((_TODOS_IQ[:, 0] == iq[0]) & (_TODOS_IQ[:, 1] == iq[1])))[0][0]
        indices.append(idx)
    return np.array(indices, dtype=np.int32)


def modular(bits: np.ndarray, A: float = 1.0) -> np.ndarray:
    """
    Modulación 16-QAM.

    Parameters
    ----------
    bits : np.ndarray int8, longitud múltiplo de 4
    A    : amplitud de escala de la constelación

    Returns
    -------
    simbolos_tx : np.ndarray float64, shape (N_simbolos, 2)  — columnas [I, Q]
    """
    n = len(bits)
    if n % 4 != 0:
        raise ValueError(f"[Modulador] Longitud de bits debe ser múltiplo de 4, tiene {n}.")

    grupos      = bits.reshape(-1, 4)
    simbolos_tx = np.zeros((len(grupos), 2), dtype=np.float64)

    for i, grupo in enumerate(grupos):
        clave = tuple(int(b) for b in grupo)
        I, Q  = _MAPA_BITS_A_IQ[clave]
        simbolos_tx[i, 0] = A * I
        simbolos_tx[i, 1] = A * Q

    return simbolos_tx


def demodular_bayes(simbolos_rx: np.ndarray, A: float = 1.0) -> np.ndarray:
    """
    Demodulador Bayesiano (Máxima Verosimilitud) para 16-QAM con AWGN.
    Con símbolos equiprobables y ruido gaussiano, equivale al vecino más cercano.

    Parameters
    ----------
    simbolos_rx : np.ndarray float64, shape (N, 2)
    A           : amplitud de la constelación (debe coincidir con la usada en modular())

    Returns
    -------
    bits_det : np.ndarray int8, shape (N*4,)
    """
    constelacion = A * _TODOS_IQ   # shape (16, 2)
    bits_det     = np.empty(len(simbolos_rx) * 4, dtype=np.int8)

    for i, rx in enumerate(simbolos_rx):
        # Distancia euclidiana al cuadrado a cada punto de la constelación
        distancias = np.sum((constelacion - rx) ** 2, axis=1)
        idx_min    = np.argmin(distancias)
        bits_det[i*4 : i*4+4] = _TODOS_BITS[idx_min]

    return bits_det


def demodular_bayes_vectorizado(simbolos_rx: np.ndarray, A: float = 1.0) -> np.ndarray:
    """
    Versión vectorizada del demodulador Bayesiano (más rápida para N grande).
    Equivalent a demodular_bayes() pero sin bucle Python.

    Returns
    -------
    bits_det : np.ndarray int8, shape (N*4,)
    """
    constelacion = (A * _TODOS_IQ)[np.newaxis, :, :]  # (1, 16, 2)
    rx_exp       = simbolos_rx[:, np.newaxis, :]       # (N, 1, 2)

    distancias   = np.sum((rx_exp - constelacion) ** 2, axis=2)  # (N, 16)
    indices_min  = np.argmin(distancias, axis=1)                  # (N,)

    bits_det = _TODOS_BITS[indices_min].reshape(-1)               # (N*4,)
    return bits_det


def calcular_amplitud(Eb_N0_dB: float, sigma: float) -> float:
    """
    Calcula la amplitud A de la constelación para que la energía por bit Eb
    sea consistente con el Eb/N0 deseado dado σ del ruido.

    Eb/N0 = Eb / (2*σ²)   →   Eb = Eb/N0 * 2 * σ²
    Eb = A² * Eb_norm       →   A = sqrt(Eb / Eb_norm)

    Parameters
    ----------
    Eb_N0_dB : float   relación Eb/N0 en dB
    sigma    : float   desviación estándar del ruido AWGN

    Returns
    -------
    A : float
    """
    Eb_N0_lineal = 10 ** (Eb_N0_dB / 10)
    Eb           = Eb_N0_lineal * 2 * sigma ** 2
    A            = np.sqrt(Eb / ENERGIA_BIT_NORM)
    return A


def simbolos_a_etiquetas(simbolos_tx: np.ndarray, A: float = 1.0) -> np.ndarray:
    """
    Convierte símbolos IQ transmitidos a etiquetas enteras (0-15).
    Útil para preparar los targets de entrenamiento de los clasificadores ML.

    Returns
    -------
    etiquetas : np.ndarray int32, shape (N,)
    """
    constelacion_norm = _TODOS_IQ  # sin escalar (las etiquetas son independientes de A)
    simbolos_norm     = simbolos_tx / A

    distancias = np.sum(
        (simbolos_norm[:, np.newaxis, :] - constelacion_norm[np.newaxis, :, :]) ** 2,
        axis=2
    )
    return np.argmin(distancias, axis=1).astype(np.int32)


def etiquetas_a_bits(etiquetas: np.ndarray) -> np.ndarray:
    """
    Convierte array de etiquetas (0-15) a bitstream.

    Returns
    -------
    bits : np.ndarray int8, shape (N*4,)
    """
    return _TODOS_BITS[etiquetas].reshape(-1)


def verificar_equiprobabilidad_simbolos(etiquetas: np.ndarray, tolerancia: float = 0.1):
    """
    Verifica que los 16 símbolos sean aproximadamente equiprobables.
    Imprime la distribución y un warning si el desvío es excesivo.
    """
    counts    = np.bincount(etiquetas, minlength=16)
    probs     = counts / len(etiquetas)
    prob_ideal = 1 / 16
    max_desvio = np.max(np.abs(probs - prob_ideal))

    print(f"[Modulador] Equiprobabilidad de símbolos — máx desvío: {max_desvio*100:.2f}%")
    if max_desvio > tolerancia:
        import warnings
        warnings.warn(
            f"[Modulador] Desvío de equiprobabilidad de símbolos ({max_desvio*100:.1f}%) "
            f"supera tolerancia ({tolerancia*100:.0f}%). Revisar fuente o cuantificación.",
            UserWarning
        )
