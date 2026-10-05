"""
canal/canal_awgn.py
===================
Modelo del canal AWGN para 16-QAM.

El canal agrega ruido gaussiano blanco aditivo con varianza σ² = N0/2.
La relación entre σ y Eb/N0 es:

    σ² = Eb / (2 * Eb/N0)

donde Eb = A² * Eb_norm es la energía promedio por bit de la constelación
escalada con amplitud A.

Referencia: Proakis & Salehi, "Digital Communications", 5th ed.
"""

import numpy as np
from canal.modulador import ENERGIA_BIT_NORM


# ---------------------------------------------------------------------------
# Cálculo de parámetros del canal
# ---------------------------------------------------------------------------

def calcular_sigma(Eb_N0_dB: float, A: float = 1.0) -> float:
    """
    Calcula la desviación estándar σ del ruido AWGN para un dado Eb/N0 y amplitud A.

    σ² = N0/2 = Eb / (2 * Eb/N0_lineal)
    Eb = A² * Eb_norm   (Eb_norm = 2.5 para 16-QAM normalizada)

    Parameters
    ----------
    Eb_N0_dB : float   Eb/N0 en dB
    A        : float   amplitud de la constelación

    Returns
    -------
    sigma : float
    """
    Eb_N0_lineal = 10 ** (Eb_N0_dB / 10)
    Eb           = (A ** 2) * ENERGIA_BIT_NORM
    sigma2       = Eb / (2 * Eb_N0_lineal)
    return float(np.sqrt(sigma2))


def calcular_amplitud_desde_sigma(Eb_N0_dB: float, sigma: float) -> float:
    """
    Función inversa: dado un σ fijo, calcula la amplitud A necesaria
    para que la constelación tenga el Eb/N0 deseado.

    A = sqrt(Eb / Eb_norm)
    Eb = 2 * σ² * Eb/N0_lineal

    Parameters
    ----------
    Eb_N0_dB : float
    sigma    : float   desviación estándar del ruido (fija)

    Returns
    -------
    A : float
    """
    Eb_N0_lineal = 10 ** (Eb_N0_dB / 10)
    Eb           = 2 * sigma ** 2 * Eb_N0_lineal
    return float(np.sqrt(Eb / ENERGIA_BIT_NORM))


# ---------------------------------------------------------------------------
# Transmisión por el canal
# ---------------------------------------------------------------------------

def agregar_ruido(simbolos_tx: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Agrega ruido gaussiano blanco aditivo a los símbolos transmitidos.

    Parameters
    ----------
    simbolos_tx : np.ndarray float64, shape (N, 2)   — [I, Q] sin ruido
    sigma       : float                               — desviación estándar del ruido
    rng         : np.random.Generator                 — generador de números aleatorios

    Returns
    -------
    simbolos_rx : np.ndarray float64, shape (N, 2)   — [I, Q] con ruido
    """
    ruido = rng.normal(0, sigma, size=simbolos_tx.shape)
    return simbolos_tx + ruido


def rotar_constelacion(simbolos: np.ndarray, angulo_rad: np.ndarray | float) -> np.ndarray:
    """
    Aplica una rotación ortogonal en el plano I/Q a los símbolos.
    
    Parameters
    ----------
    simbolos   : np.ndarray float64, shape (N, 2)
    angulo_rad : float o np.ndarray de shape (N,)
                 Si es escalar: rotación rígida constante (S2).
                 Si es array (N,): rotación estocástica independiente por símbolo (S3).

    Returns
    -------
    simbolos_rot : np.ndarray float64, shape (N, 2)
    """
    cos_th = np.cos(angulo_rad)
    sin_th = np.sin(angulo_rad)
    I_rot = simbolos[:, 0] * cos_th - simbolos[:, 1] * sin_th
    Q_rot = simbolos[:, 0] * sin_th + simbolos[:, 1] * cos_th
    return np.column_stack((I_rot, Q_rot))


def aplicar_error_fase_estatico(simbolos_tx: np.ndarray, phi_e_deg: float) -> np.ndarray:
    """
    Aplica una desalineación angular de portadora estática constante phi_e_deg [Escenario S2].
    """
    if phi_e_deg == 0.0:
        return simbolos_tx
    return rotar_constelacion(simbolos_tx, np.radians(phi_e_deg))


def aplicar_ruido_fase_estocastico(
    simbolos_tx: np.ndarray,
    sigma_phi_deg: float,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica ruido de fase estocástico gaussiano i.i.d. por símbolo [Escenario S3].
    
    phi_k ~ N(0, sigma_phi^2)

    Returns
    -------
    simbolos_rot : np.ndarray float64, shape (N, 2)
    phi_k_rad    : np.ndarray float64, shape (N,) — realizaciones de fase aplicadas
    """
    if sigma_phi_deg <= 0.0:
        return simbolos_tx, np.zeros(len(simbolos_tx), dtype=np.float64)
    phi_k = rng.normal(0.0, np.radians(sigma_phi_deg), size=len(simbolos_tx))
    simbolos_rot = rotar_constelacion(simbolos_tx, phi_k)
    return simbolos_rot, phi_k


def transmitir(
    simbolos_tx:   np.ndarray,
    Eb_N0_dB:      float,
    A:             float,
    rng:           np.random.Generator,
    phi_e_deg:     float = 0.0,
    sigma_phi_deg: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Pipeline completo del canal:
      1. Aplica error de fase estático phi_e_deg (si != 0) [Escenario S2]
      2. Aplica ruido de fase estocástico sigma_phi_deg (si != 0) [Escenario S3]
      3. Agrega ruido térmico AWGN según Eb_N0_dB y amplitud A [Escenarios S1, S2, S3]

    Si phi_e_deg=0 y sigma_phi_deg=0, reproduce idénticamente el canal AWGN puro (S1).

    Parameters
    ----------
    simbolos_tx   : np.ndarray, shape (N, 2)
    Eb_N0_dB      : float   — punto del barrido
    A             : float   — amplitud de la constelación
    rng           : np.random.Generator
    phi_e_deg     : float   — error de fase estático en grados (default: 0.0)
    sigma_phi_deg : float   — desviación estándar de ruido de fase en grados (default: 0.0)

    Returns
    -------
    simbolos_rx : np.ndarray, shape (N, 2)
    sigma       : float   — sigma del ruido térmico AWGN usado
    """
    simbolos_mod = simbolos_tx
    if phi_e_deg != 0.0:
        simbolos_mod = aplicar_error_fase_estatico(simbolos_mod, phi_e_deg)
    if sigma_phi_deg > 0.0:
        simbolos_mod, _ = aplicar_ruido_fase_estocastico(simbolos_mod, sigma_phi_deg, rng)

    sigma       = calcular_sigma(Eb_N0_dB, A)
    simbolos_rx = agregar_ruido(simbolos_mod, sigma, rng)
    return simbolos_rx, sigma


# ---------------------------------------------------------------------------
# Curva teórica de BER para 16-QAM
# ---------------------------------------------------------------------------

def ber_teorica_16qam(Eb_N0_dB_array) -> np.ndarray:
    """
    Probabilidad de error de bit teórica para 16-QAM con detección coherente
    en canal AWGN.

    Pb = (3/4) * (1/2) * erfc( sqrt( (4/5) * Eb/N0 / sqrt(2) ) )

    Formulación según Proakis & Salehi (2008), también usada en los
    trabajos de referencia del proyecto.

    Parameters
    ----------
    Eb_N0_dB_array : array-like de valores Eb/N0 en dB

    Returns
    -------
    Pb : np.ndarray
    """
    from scipy.special import erfc
    Eb_N0_dB  = np.asarray(Eb_N0_dB_array, dtype=np.float64)
    Eb_N0_lin = 10 ** (Eb_N0_dB / 10)
    Pb        = (3 / 4) * 0.5 * erfc(np.sqrt((4 / 5) * Eb_N0_lin) / np.sqrt(2))
    return Pb
