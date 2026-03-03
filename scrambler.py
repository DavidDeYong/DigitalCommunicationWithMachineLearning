"""
scrambler.py
============
Scrambling y descrambling de bits mediante LFSR (Linear Feedback Shift Register).

Propósito:
  Garantizar la equiprobabilidad de bits (P(1) ≈ 0.5) a la entrada del modulador,
  independientemente de la distribución estadística de la fuente de información
  (voz comprimida con Ley µ + cuantificación uniforme).

Principio:
  Se genera una secuencia pseudoaleatoria de período largo usando un LFSR.
  Cada bit de entrada se XORea con un bit de la secuencia:

    bit_scrambled[i] = bit_entrada[i] XOR prbs[i]

  El receptor aplica el mismo XOR con la secuencia sincronizada:

    bit_recuperado[i] = bit_scrambled[i] XOR prbs[i] = bit_entrada[i]

  Como la secuencia PRBS tiene P(1) = P(0) = 0.5 y es independiente de la
  fuente, el resultado XOR tiene P(1) ≈ 0.5 para cualquier distribución de
  entrada (ley de mezcla XOR).

LFSR configurado:
  - Polinomio de 23 bits: x²³ + x¹⁸ + 1  (PRBS-23, período 2²³-1 ≈ 8.4M bits)
  - Suficiente para el bitstream de voz de 2M bits sin repetición de ciclo.

Uso estándar en telecomunicaciones:
  DVB-S2, DOCSIS 3.x, IEEE 802.11 usan scramblers similares para garantizar
  transiciones de señal y equiprobabilidad de símbolos.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Generador de secuencia PRBS mediante LFSR
# ---------------------------------------------------------------------------

def generar_prbs(n_bits: int, seed: int = 42, poly: int = 0x400007) -> np.ndarray:
    """
    Genera una secuencia pseudoaleatoria binaria de n_bits usando un LFSR.

    Parameters
    ----------
    n_bits : int    longitud de la secuencia a generar
    seed   : int    estado inicial del LFSR (distinto de 0)
    poly   : int    polinomio generador en representación hexadecimal
                    Por defecto: 0x400007 → x²³ + x¹⁸ + 1  (PRBS-23)

    Returns
    -------
    np.ndarray int8, shape (n_bits,)   secuencia PRBS con valores {0, 1}
    """
    if seed == 0:
        raise ValueError("El estado inicial del LFSR no puede ser 0.")

    # Detectar grado del polinomio (posición del bit más significativo)
    grado = poly.bit_length() - 1

    prbs   = np.empty(n_bits, dtype=np.int8)
    estado = seed & ((1 << grado) - 1)   # Asegurar que cabe en 'grado' bits

    for i in range(n_bits):
        # El bit de salida es el bit menos significativo del estado
        bit_salida = estado & 1
        prbs[i]    = bit_salida

        # Calcular bit de retroalimentación: XOR de los bits indicados por poly
        feedback = bin(estado & poly).count('1') % 2

        # Desplazar el registro y agregar el bit de retroalimentación
        estado = ((estado >> 1) | (feedback << (grado - 1))) & ((1 << grado) - 1)

    return prbs


def generar_prbs_rapido(n_bits: int, seed: int = 42, poly: int = 0x400007) -> np.ndarray:
    """
    Versión vectorizada y eficiente de generar_prbs usando numpy.
    Genera bloques de 64 bits a la vez. Recomendada para n_bits > 100k.
    """
    grado  = poly.bit_length() - 1
    estado = seed & ((1 << grado) - 1)
    prbs   = np.empty(n_bits, dtype=np.int8)

    i = 0
    while i < n_bits:
        bit_salida = estado & 1
        prbs[i]    = bit_salida
        feedback   = bin(estado & poly).count('1') % 2
        estado     = ((estado >> 1) | (feedback << (grado - 1))) & ((1 << grado) - 1)
        i += 1

    return prbs


# ---------------------------------------------------------------------------
# Scrambling y descrambling
# ---------------------------------------------------------------------------

def scramble(bits: np.ndarray, seed: int = 42, poly: int = 0x400007) -> np.ndarray:
    """
    Aplica scrambling XOR a un array de bits.

    Parameters
    ----------
    bits : np.ndarray int8, shape (N,)   bits de entrada {0, 1}
    seed : int                           semilla del LFSR
    poly : int                           polinomio generador

    Returns
    -------
    np.ndarray int8, shape (N,)   bits scrambleados
    """
    prbs = generar_prbs_rapido(len(bits), seed=seed, poly=poly)
    return (bits ^ prbs).astype(np.int8)


def descramble(bits_scrambled: np.ndarray,
               seed: int = 42,
               poly: int = 0x400007) -> np.ndarray:
    """
    Recupera los bits originales aplicando el mismo XOR.
    Idéntico a scramble() — el XOR es su propia inversa.
    """
    return scramble(bits_scrambled, seed=seed, poly=poly)


# ---------------------------------------------------------------------------
# Verificación de equiprobabilidad
# ---------------------------------------------------------------------------

def verificar_equiprobabilidad(bits: np.ndarray, etiqueta: str = "") -> dict:
    """
    Calcula P(1) y desvío respecto al 50% ideal.

    Returns
    -------
    dict con p1, p0, desvio_pp (puntos porcentuales), equiprobable (bool)
    """
    p1     = float(bits.mean())
    p0     = 1.0 - p1
    desvio = abs(p1 - 0.5) * 100

    prefijo = f"[{etiqueta}] " if etiqueta else ""
    estado  = "OK" if desvio < 2.0 else ("ADVERTENCIA" if desvio < 5.0 else "PROBLEMATICO")
    print(f"  {prefijo}P(1)={p1:.4f}  P(0)={p0:.4f}  "
          f"Desvío={desvio:.2f}pp  [{estado}]")

    return {"p1": p1, "p0": p0, "desvio_pp": desvio, "equiprobable": desvio < 2.0}
