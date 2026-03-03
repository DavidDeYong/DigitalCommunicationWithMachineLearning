"""
fuente/fuente.py
================
Módulo de fuente de información.

Responsabilidades:
  - Leer archivo de audio WAV (mono, canal único)
  - Fallback: generar señal sintética si no existe el archivo
  - Aplicar compresión Ley µ
  - Cuantificar uniformemente tras la compresión
  - Codificar cada muestra en N bits → bitstream
  - Verificar equiprobabilidad de bits (validación de fuente)
"""

import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd
import warnings


# ---------------------------------------------------------------------------
# Lectura de audio
# ---------------------------------------------------------------------------

def leer_audio(ruta: str, fs_objetivo: int) -> tuple[np.ndarray, int]:
    """
    Lee un archivo WAV y lo convierte a mono con la frecuencia de muestreo objetivo.

    Returns
    -------
    audio : np.ndarray  float64 normalizado en [-1, 1]
    fs    : int         frecuencia de muestreo real del archivo
    """
    fs, data = wavfile.read(ruta)

    # Convertir a mono si es estéreo
    if data.ndim > 1:
        data = data[:, 0]

    # Normalizar a float64 en [-1, 1]
    audio = data.astype(np.float64)
    audio /= np.max(np.abs(audio)) + 1e-12

    # Resamplear si la fs no coincide con la objetivo
    if fs != fs_objetivo:
        factor = gcd(fs_objetivo, fs)
        up   = fs_objetivo // factor
        down = fs // factor
        audio = resample_poly(audio, up, down)

    return audio, fs_objetivo


def generar_audio_sintetico(fs: int, duracion_s: float = 10.0, seed: int = 42) -> np.ndarray:
    """
    Genera una señal de audio sintética como fallback cuando no existe el archivo WAV.
    Consiste en una suma de senoidales a frecuencias típicas de la voz humana,
    más ruido gaussiano suave para evitar la simetría perfecta.

    Returns
    -------
    audio : np.ndarray  float64 normalizado en [-1, 1]
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duracion_s, int(fs * duracion_s), endpoint=False)

    # Frecuencias características de la voz (Hz) y amplitudes relativas
    frecuencias  = [120, 240, 500, 800, 1200, 2000, 3000]
    amplitudes   = [0.6,  0.5, 0.4, 0.3,  0.25,  0.15, 0.10]
    fases        = rng.uniform(0, 2 * np.pi, len(frecuencias))

    audio = sum(a * np.sin(2 * np.pi * f * t + phi)
                for f, a, phi in zip(frecuencias, amplitudes, fases))

    # Ruido gaussiano suave
    audio += 0.05 * rng.standard_normal(len(t))

    # Normalizar
    audio /= np.max(np.abs(audio)) + 1e-12
    return audio


def obtener_audio(ruta: str, fs_objetivo: int, seed: int = 42) -> tuple[np.ndarray, int, bool]:
    """
    Intenta cargar el audio desde 'ruta'. Si no existe, usa el fallback sintético.

    Returns
    -------
    audio    : np.ndarray
    fs       : int
    sintetico: bool   True si se usó el fallback
    """
    try:
        audio, fs = leer_audio(ruta, fs_objetivo)
        print(f"[Fuente] Audio cargado desde '{ruta}' | fs={fs} Hz | {len(audio)} muestras")
        return audio, fs, False
    except FileNotFoundError:
        warnings.warn(
            f"\n[Fuente] ADVERTENCIA: No se encontró '{ruta}'.\n"
            f"  → Usando señal de audio SINTÉTICA como fallback.\n"
            f"  → Para resultados del artículo, proveer el archivo WAV real.\n",
            UserWarning
        )
        audio = generar_audio_sintetico(fs_objetivo, seed=seed)
        print(f"[Fuente] Audio sintético generado | fs={fs_objetivo} Hz | {len(audio)} muestras")
        return audio, fs_objetivo, True


# ---------------------------------------------------------------------------
# Ley µ
# ---------------------------------------------------------------------------

def comprimir_ley_mu(x: np.ndarray, mu: float = 255) -> np.ndarray:
    """
    Compresión Ley µ. Entrada x debe estar normalizada en [-1, 1].
    F(x) = sign(x) * log(1 + µ|x|) / log(1 + µ)
    """
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def expandir_ley_mu(y: np.ndarray, mu: float = 255) -> np.ndarray:
    """
    Expansión (decodificación) Ley µ.
    x(y) = sign(y) * [(1+µ)^|y| - 1] / µ
    """
    return np.sign(y) * (np.power(1 + mu, np.abs(y)) - 1) / mu


# ---------------------------------------------------------------------------
# Cuantificación uniforme
# ---------------------------------------------------------------------------

def cuantificar(audio_comprimido: np.ndarray, n_bits: int) -> tuple[np.ndarray, float, float]:
    """
    Cuantificación uniforme de la señal comprimida.

    Parameters
    ----------
    audio_comprimido : señal en [-1, 1] tras compresión Ley µ
    n_bits           : bits de cuantificación (e.g., 8 → 256 niveles)

    Returns
    -------
    indices   : np.ndarray int   índice de nivel para cada muestra (0 a 2^n_bits - 1)
    x_min     : float            valor mínimo (para reconstrucción)
    x_max     : float            valor máximo (para reconstrucción)
    """
    niveles = 2 ** n_bits
    x_min   = audio_comprimido.min()
    x_max   = audio_comprimido.max()
    rango   = x_max - x_min + 1e-12

    indices = np.floor((audio_comprimido - x_min) / rango * niveles).astype(int)
    indices = np.clip(indices, 0, niveles - 1)
    return indices, x_min, x_max


def indices_a_bitstream(indices: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Convierte array de índices enteros a un bitstream flat (numpy int8).
    Cada índice se representa con n_bits bits (MSB primero).
    """
    bits_list = []
    for idx in indices:
        for b in range(n_bits - 1, -1, -1):
            bits_list.append((int(idx) >> b) & 1)
    return np.array(bits_list, dtype=np.int8)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def procesar_fuente(
    ruta_audio:  str,
    mu:          float,
    bits_cuant:  int,
    fs_objetivo: int,
    seed:        int = 42
) -> dict:
    """
    Pipeline completo de la fuente de información.

    Returns
    -------
    dict con:
        'bitstream'  : np.ndarray int8  — secuencia completa de bits
        'n_simbolos' : int              — número de símbolos 16-QAM disponibles
        'audio_raw'  : np.ndarray       — audio original normalizado
        'sintetico'  : bool             — True si se usó fallback
        'fs'         : int
        'x_min'      : float            — para reconstrucción
        'x_max'      : float
        'mu'         : float
        'bits_cuant' : int
    """
    # 1. Obtener audio
    audio, fs, sintetico = obtener_audio(ruta_audio, fs_objetivo, seed)

    # 2. Comprimir con Ley µ
    audio_comprimido = comprimir_ley_mu(audio, mu)

    # 3. Cuantificar
    indices, x_min, x_max = cuantificar(audio_comprimido, bits_cuant)

    # 4. Generar bitstream
    bitstream = indices_a_bitstream(indices, bits_cuant)

    # 5. Calcular cuántos símbolos 16-QAM (4 bits cada uno) hay disponibles
    bits_por_simbolo = 4   # 16-QAM
    n_simbolos = len(bitstream) // bits_por_simbolo

    # Truncar bitstream a múltiplo exacto de bits_por_simbolo
    bitstream = bitstream[:n_simbolos * bits_por_simbolo]

    # 6. Validar equiprobabilidad
    _validar_equiprobabilidad(bitstream)

    print(f"[Fuente] Bitstream: {len(bitstream)} bits | {n_simbolos} símbolos 16-QAM disponibles")

    return {
        'bitstream'  : bitstream,
        'n_simbolos' : n_simbolos,
        'audio_raw'  : audio,
        'sintetico'  : sintetico,
        'fs'         : fs,
        'x_min'      : x_min,
        'x_max'      : x_max,
        'mu'         : mu,
        'bits_cuant' : bits_cuant,
    }


def _validar_equiprobabilidad(bitstream: np.ndarray, umbral: float = 0.05):
    """
    Verifica que la proporción de 0s y 1s esté dentro del umbral respecto al 50%.
    Imprime un warning si el desvío es mayor al umbral.
    """
    p_unos  = np.mean(bitstream)
    p_ceros = 1 - p_unos
    desvio  = abs(p_unos - 0.5)

    print(f"[Fuente] Equiprobabilidad → P(1)={p_unos:.4f}  P(0)={p_ceros:.4f}  "
          f"Desvío del 50%: {desvio*100:.2f}%")

    if desvio > umbral:
        warnings.warn(
            f"[Fuente] ADVERTENCIA: Desvío de equiprobabilidad ({desvio*100:.1f}%) "
            f"supera el umbral ({umbral*100:.0f}%). "
            f"Considerar ajustar el parámetro µ.",
            UserWarning
        )


def extraer_segmento(
    fuente_dict:     dict,
    inicio_simbolo:  int,
    n_simbolos:      int,
    usar_scrambling: bool = False,
    scrambling_seed: int  = 42,
    scrambling_poly: int  = 0x400007,
) -> np.ndarray:
    """
    Extrae un segmento del bitstream como array de bits.

    Si usar_scrambling=True aplica XOR con una secuencia PRBS,
    garantizando P(1) ≈ 0.5 independientemente de la distribución
    de la fuente. El descrambling en recepción es idéntico (mismo XOR).

    Parameters
    ----------
    inicio_simbolo  : índice del primer símbolo a extraer
    n_simbolos      : cantidad de símbolos a extraer
    usar_scrambling : si True, aplica PRBS-XOR scrambling
    scrambling_seed : semilla del LFSR
    scrambling_poly : polinomio generador del LFSR

    Returns
    -------
    np.ndarray int8  de longitud n_simbolos * 4
    """
    bps   = 4  # bits por símbolo 16-QAM
    start = inicio_simbolo * bps
    end   = start + n_simbolos * bps
    total = fuente_dict['n_simbolos'] * bps

    if end > total:
        raise ValueError(
            f"[Fuente] Segmento solicitado ({inicio_simbolo}+{n_simbolos} símbolos) "
            f"excede el bitstream disponible ({fuente_dict['n_simbolos']} símbolos)."
        )

    bits = fuente_dict['bitstream'][start:end].copy()

    if usar_scrambling:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scrambler import scramble
        bits = scramble(bits, seed=scrambling_seed, poly=scrambling_poly)

    return bits
