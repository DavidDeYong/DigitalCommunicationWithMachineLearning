"""
benchmark_elm_init.py
=====================
Compara las cuatro estrategias de inicialización del ELM en el canal AWGN
para la demodulación 16-QAM.

Protocolo
---------
- N_train = 100.000 símbolos sintéticos equiprobables (secuencia piloto)
- N_test  = 100.000 símbolos de voz real con scrambling
- Eb/N0   = {0, 2, 4, 6, 8, 10, 12} dB
- 10 corridas independientes por estrategia (seeds estándar del benchmark)
- Hiperparámetros: L=500, C=100, tanh (óptimos del benchmark para 'aleatorio')
  → para las otras estrategias se permite re-optimizar si REOPTIMIZAR=True

Uso
---
    python benchmark_elm_init.py              # usa hiperparámetros fijos
    python benchmark_elm_init.py --reopt      # re-optimiza cada estrategia
    python benchmark_elm_init.py --sigma 0.3  # cambia dispersión constelación
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import erfc
from scipy import stats
from collections import defaultdict

# ── Ajustar path para importar módulos del proyecto ──────────────────────────
# Modificar esta ruta según la ubicación de tu proyecto:
PROYECTO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROYECTO_DIR)

# Importaciones del proyecto
# Si el archivo está fuera de la carpeta de clasificadores, ajustar:
try:
    from clasificadores.elm_clf import _ELMCore, _CONSTELACION_16QAM
    USAR_MODULO = True
except ImportError:
    # Fallback: definiciones locales (para correr como script independiente)
    USAR_MODULO = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 137, 256, 512, 1024, 2048, 3141, 7777, 9999, 31416]
EB_N0_DB = [0, 2, 4, 6, 8, 10, 12]
N_TRAIN  = 100_000
N_TEST   = 100_000
Eb       = 2.5       # J/bit

# Hiperparámetros fijos (óptimos del benchmark para init='aleatorio')
L_DEFAULT   = 500
C_DEFAULT   = 100.0
ACT_DEFAULT = 'tanh'

# Estrategias a comparar
ESTRATEGIAS = ['aleatorio', 'constelacion', 'voronoi', 'hibrido']

COLORES = {
    'aleatorio'   : '#E53935',   # rojo
    'constelacion': '#1E88E5',   # azul
    'voronoi'     : '#43A047',   # verde
    'hibrido'     : '#FB8C00',   # naranja
}
MARKERS = {
    'aleatorio'   : 'o',
    'constelacion': 's',
    'voronoi'     : '^',
    'hibrido'     : 'D',
}

DIR_SALIDA = os.path.join(PROYECTO_DIR, "resultados_elm_init")
os.makedirs(DIR_SALIDA, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRÍA 16-QAM
# ─────────────────────────────────────────────────────────────────────────────

CONSTELACION = np.array([
    [-3,-3],[-3,-1],[-3, 1],[-3, 3],
    [-1,-3],[-1,-1],[-1, 1],[-1, 3],
    [ 1,-3],[ 1,-1],[ 1, 1],[ 1, 3],
    [ 3,-3],[ 3,-1],[ 3, 1],[ 3, 3],
], dtype=np.float64)

FRONTERAS_W = np.array([
    [ 1, 0],[-1, 0],[ 1, 0],[-1, 0],
    [ 0, 1],[ 0,-1],[ 0, 1],[ 0,-1],
], dtype=np.float64)
FRONTERAS_B = np.array([0, 0, -2, 2, 0, 0, -2, 2], dtype=np.float64)

# Tabla Gray 16-QAM: símbolo (0-15) → 4 bits [I_MSB, I_LSB, Q_MSB, Q_LSB]
# I_idx = símbolo // 4, Q_idx = símbolo % 4
# Código Gray 2-bit por componente: idx ^ (idx >> 1)
def _build_gray_table():
    t = np.zeros((16, 4), dtype=np.int32)
    for i in range(16):
        Ig = (i // 4) ^ ((i // 4) >> 1)
        Qg = (i %  4) ^ ((i %  4) >> 1)
        t[i] = [(Ig >> 1) & 1, Ig & 1, (Qg >> 1) & 1, Qg & 1]
    return t

GRAY_TABLE = _build_gray_table()

def ber_desde_simbolos(y_pred, y_true):
    """
    BER correcta para 16-QAM con mapeo Gray.
    Cuenta errores de bit individuales (no errores de símbolo).
    Para 16-QAM: 4 bits/símbolo → BER = bit_errors / (N * 4).
    """
    bits_pred = GRAY_TABLE[y_pred]   # (N, 4)
    bits_true = GRAY_TABLE[y_true]   # (N, 4)
    return np.sum(bits_pred != bits_true) / (len(y_pred) * 4)


# ─────────────────────────────────────────────────────────────────────────────
# NÚCLEO ELM AUTÓNOMO (si no se puede importar el módulo)
# ─────────────────────────────────────────────────────────────────────────────

def activacion(H, func):
    if func == 'tanh':    return np.tanh(H)
    if func == 'relu':    return np.maximum(0, H)
    if func == 'sigmoid': return 1/(1+np.exp(-np.clip(H,-500,500)))
    raise ValueError(func)

def one_hot(y, n):
    Y = np.zeros((len(y), n)); Y[np.arange(len(y)), y] = 1; return Y

class ELM:
    def __init__(self, L, act, C, seed, strategy, sigma=0.5):
        self.L = L; self.act = act; self.C = C
        self.seed = seed; self.strategy = strategy; self.sigma = sigma

    def _init(self, rng):
        if self.strategy == 'aleatorio':
            self.W = rng.standard_normal((self.L, 2))
            self.b = rng.standard_normal(self.L)
        elif self.strategy == 'constelacion':
            idx    = np.arange(self.L) % 16
            self.W = CONSTELACION[idx] + rng.normal(0, self.sigma, (self.L, 2))
            self.b = rng.normal(0, self.sigma, self.L)
        elif self.strategy == 'voronoi':
            idx    = np.arange(self.L) % len(FRONTERAS_W)
            self.W = FRONTERAS_W[idx] + rng.normal(0, 0.05, (self.L, 2))
            self.b = FRONTERAS_B[idx] + rng.normal(0, 0.05, self.L)
        elif self.strategy == 'hibrido':
            L2     = self.L // 2
            idx    = np.arange(L2) % 16
            Wg     = CONSTELACION[idx] + rng.normal(0, self.sigma, (L2, 2))
            bg     = rng.normal(0, self.sigma, L2)
            Wr     = rng.standard_normal((self.L-L2, 2))
            br     = rng.standard_normal(self.L-L2)
            self.W = np.vstack([Wg, Wr])
            self.b = np.concatenate([bg, br])

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self._init(rng)
        H = activacion(X @ self.W.T + self.b, self.act)
        Y = one_hot(y.astype(int), int(y.max())+1)
        A = H.T @ H + np.eye(self.L)/self.C
        self.beta = np.linalg.solve(A, H.T @ Y)

    def predict(self, X):
        H = activacion(X @ self.W.T + self.b, self.act)
        return np.argmax(H @ self.beta, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def sigma2_awgn(eb_n0_db):
    return Eb / (2 * 10**(eb_n0_db / 10))

def generar_tren(n, seed_tren):
    """N símbolos equiprobables + ruido @ 8 dB (para hiperparámetros)."""
    rng = np.random.default_rng(seed_tren)
    y   = rng.integers(0, 16, n)
    X   = CONSTELACION[y] + rng.normal(0, np.sqrt(sigma2_awgn(8)), (n, 2))
    return X.astype(np.float64), y.astype(np.int32)

def generar_test(n, eb_n0_db, seed_canal):
    """N símbolos de voz simulada (equiprobable por scrambling) + ruido."""
    rng     = np.random.default_rng(seed_canal)
    y       = rng.integers(0, 16, n)   # simula scrambling perfecto
    sigma   = np.sqrt(sigma2_awgn(eb_n0_db))
    X       = CONSTELACION[y] + rng.normal(0, sigma, (n, 2))
    return X.astype(np.float64), y.astype(np.int32)

def ber_teorica(eb_n0_db):
    lin = 10**(eb_n0_db / 10)
    return 0.75 * 0.5 * erfc(np.sqrt(0.8 * lin) / np.sqrt(2))


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def correr_benchmark(sigma_init=0.5, L=L_DEFAULT, C=C_DEFAULT, act=ACT_DEFAULT):
    """
    Corre las 4 estrategias × 10 seeds × 7 puntos Eb/N0.
    Devuelve dict[estrategia][eb_n0] = lista de BER.
    """
    resultados = defaultdict(lambda: defaultdict(list))
    tiempos    = defaultdict(list)

    SEED_TREN = 2025   # semilla fija para el conjunto de entrenamiento

    print("=" * 70)
    print(f"  BENCHMARK ELM — Estrategias de inicialización")
    print(f"  L={L}  C={C}  act={act}  sigma_init={sigma_init}")
    print(f"  N_train={N_TRAIN}  N_test={N_TEST}  n_seeds={len(SEEDS)}")
    print("=" * 70)

    X_tren, y_tren = generar_tren(N_TRAIN, SEED_TREN)

    for estrategia in ESTRATEGIAS:
        print(f"\n── Estrategia: {estrategia.upper()} ──")
        for seed_idx, seed in enumerate(SEEDS):
            t0  = time.perf_counter()
            elm = ELM(L, act, C, seed=SEED_TREN,
                      strategy=estrategia, sigma=sigma_init)
            elm.fit(X_tren, y_tren)
            t_train = time.perf_counter() - t0
            tiempos[estrategia].append(t_train)

            for eb in EB_N0_DB:
                X_test, y_test = generar_test(N_TEST, eb, seed)
                y_pred = elm.predict(X_test)
                ber    = ber_desde_simbolos(y_pred, y_test)
                resultados[estrategia][eb].append(float(ber))

            print(f"  seed={seed:>6}  t_train={t_train:.3f}s  "
                  f"BER@10dB={resultados[estrategia][10][-1]:.4e}")

    return resultados, tiempos


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────

def analizar(resultados):
    t_val = stats.t.ppf(0.975, df=len(SEEDS)-1)  # t_{0.025, 9} = 2.262
    print(f"\n{'='*75}")
    print(f"  RESUMEN ESTADÍSTICO  (t_{{0.025,{len(SEEDS)-1}}} = {t_val:.4f})")
    print(f"{'='*75}")

    # Tabla BER @ 10 dB
    print(f"\n  BER media @ 10 dB por estrategia:")
    print(f"  {'Estrategia':<15}  {'BER media':>12}  {'±IC95%':>12}  "
          f"{'CV%':>7}  {'vs aleatorio':>14}")
    print("  " + "-"*65)
    ber_base = np.mean(resultados['aleatorio'][10])
    for est in ESTRATEGIAS:
        v   = np.array(resultados[est][10])
        m   = np.mean(v)
        ic  = t_val * np.std(v, ddof=1) / np.sqrt(len(v))
        cv  = np.std(v, ddof=1) / m * 100
        d   = 10 * np.log10(m / ber_base)
        print(f"  {est:<15}  {m:>12.4e}  {ic:>12.2e}  {cv:>7.2f}  {d:>+14.4f} dB")

    # Test t pareado vs 'aleatorio' @ 10 dB
    print(f"\n  Test t pareado vs 'aleatorio' @ 10 dB:")
    v_base = np.array(resultados['aleatorio'][10])
    for est in ESTRATEGIAS:
        if est == 'aleatorio': continue
        v   = np.array(resultados[est][10])
        d   = v - v_base
        t, p = stats.ttest_1samp(d, 0)
        sig  = "* (significativo)" if p < 0.05 else "  (no significativo)"
        print(f"  {est:<15}  p={p:.4f}  {sig}")

    # BER completa por Eb/N0
    print(f"\n  BER media por estrategia y Eb/N0:")
    header = f"  {'Estrategia':<15}" + "".join(f"  {eb:2d}dB" for eb in EB_N0_DB)
    print(header)
    print("  " + "-"*70)
    print(f"  {'Teórica':<15}" +
          "".join(f"  {ber_teorica(eb):.3e}" for eb in EB_N0_DB))
    print("  " + "-"*70)
    for est in ESTRATEGIAS:
        row = f"  {est:<15}"
        for eb in EB_N0_DB:
            row += f"  {np.mean(resultados[est][eb]):.3e}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────────────────────

def graficar(resultados, tiempos, sigma_init):
    t_val = stats.t.ppf(0.975, df=len(SEEDS)-1)

    # ── Fig 1: BER vs Eb/N0 con IC 95% ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    # Curva teórica
    eb_fine = np.linspace(0, 12, 200)
    ax.semilogy(eb_fine, [ber_teorica(e) for e in eb_fine],
                'k--', lw=1.5, label='Teórica', alpha=0.7)

    for est in ESTRATEGIAS:
        bers_media = [np.mean(resultados[est][eb]) for eb in EB_N0_DB]
        bers_ic    = [t_val * np.std(resultados[est][eb], ddof=1)
                      / np.sqrt(len(SEEDS)) for eb in EB_N0_DB]
        ax.errorbar(EB_N0_DB, bers_media,
                    yerr=bers_ic,
                    fmt=f'-{MARKERS[est]}',
                    color=COLORES[est],
                    ms=7, lw=1.8, capsize=4,
                    label=f'ELM-{est}')

    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=12)
    ax.set_ylabel('BER media', fontsize=12)
    ax.set_title('BER vs $E_b/N_0$ — Estrategias de inicialización ELM\n'
                 f'Media ± IC 95% ({len(SEEDS)} corridas)', fontsize=11)
    ax.set_xticks(EB_N0_DB)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    ruta = os.path.join(DIR_SALIDA, 'elm_init_ber_vs_ebn0.png')
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"\n  [OK] {os.path.basename(ruta)}")

    # ── Fig 2: Degradación en dB vs 'aleatorio' ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bers_base = np.array([np.mean(resultados['aleatorio'][eb])
                          for eb in EB_N0_DB])

    for est in ESTRATEGIAS:
        if est == 'aleatorio': continue
        bers = np.array([np.mean(resultados[est][eb]) for eb in EB_N0_DB])
        delta = 10 * np.log10(bers / bers_base)
        ax.plot(EB_N0_DB, delta,
                f'-{MARKERS[est]}',
                color=COLORES[est],
                ms=7, lw=1.8,
                label=f'ELM-{est}')

    ax.axhline(0, color='red', ls='--', lw=1.2, alpha=0.7,
               label='ELM-aleatorio (referencia)')
    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=12)
    ax.set_ylabel(r'$\Delta$ BER vs aleatorio (dB)', fontsize=12)
    ax.set_title('Degradación/Mejora respecto a ELM-aleatorio\n'
                 r'$\Delta < 0$ = mejor que aleatorio', fontsize=11)
    ax.set_xticks(EB_N0_DB)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    ruta = os.path.join(DIR_SALIDA, 'elm_init_degradacion.png')
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")

    # ── Fig 3: Boxplot BER @ 10 dB por estrategia ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data_box = [resultados[est][10] for est in ESTRATEGIAS]
    colores_box = [COLORES[est] for est in ESTRATEGIAS]

    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker='x', ms=5, alpha=0.5))
    for patch, color in zip(bp['boxes'], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(ber_teorica(10), color='black', ls='--', lw=1,
               alpha=0.6, label='BER teórica @ 10 dB')
    ax.set_xticks(range(1, len(ESTRATEGIAS)+1))
    ax.set_xticklabels([f'ELM\n{e}' for e in ESTRATEGIAS], fontsize=9)
    ax.set_ylabel('BER @ 10 dB', fontsize=11)
    ax.set_title(f'Distribución de BER @ 10 dB — {len(SEEDS)} corridas\n'
                 f'(L={L_DEFAULT}, C={C_DEFAULT}, {ACT_DEFAULT})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    fig.tight_layout()
    ruta = os.path.join(DIR_SALIDA, 'elm_init_boxplot_ber10.png')
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")

    # ── Fig 4: CV% por estrategia y Eb/N0 ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for est in ESTRATEGIAS:
        cvs = []
        for eb in EB_N0_DB:
            v  = np.array(resultados[est][eb])
            cv = np.std(v, ddof=1) / np.mean(v) * 100
            cvs.append(cv)
        ax.plot(EB_N0_DB, cvs,
                f'-{MARKERS[est]}',
                color=COLORES[est],
                ms=6, lw=1.5,
                label=f'ELM-{est}')

    ax.set_xlabel(r'$E_b/N_0$ (dB)', fontsize=12)
    ax.set_ylabel('CV% (desviación/media × 100)', fontsize=11)
    ax.set_title('Consistencia entre corridas (CV%) — Estrategias ELM', fontsize=11)
    ax.set_xticks(EB_N0_DB)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    ruta = os.path.join(DIR_SALIDA, 'elm_init_cv.png')
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")

    print(f"\n  Figuras guardadas en: {DIR_SALIDA}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark estrategias de inicialización ELM para 16-QAM')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Dispersión alrededor de la constelación (default: 0.5)')
    parser.add_argument('--L',    type=int,   default=L_DEFAULT,
                        help=f'Nodos ocultos (default: {L_DEFAULT})')
    parser.add_argument('--C',    type=float, default=C_DEFAULT,
                        help=f'Regularización (default: {C_DEFAULT})')
    parser.add_argument('--act',  type=str,   default=ACT_DEFAULT,
                        choices=['tanh','relu','sigmoid'],
                        help=f'Activación (default: {ACT_DEFAULT})')
    args = parser.parse_args()

    resultados, tiempos = correr_benchmark(
        sigma_init = args.sigma,
        L          = args.L,
        C          = args.C,
        act        = args.act,
    )
    analizar(resultados)
    graficar(resultados, tiempos, args.sigma)


if __name__ == '__main__':
    main()
