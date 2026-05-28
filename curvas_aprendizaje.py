"""
curvas_aprendizaje.py
=====================
Experimento de curvas de aprendizaje: BER vs. N_train para cada clasificador.

Pregunta que responde
---------------------
  ¿Cuántos símbolos de trama piloto necesita cada clasificador para alcanzar
  su BER asintótica? Esta es la métrica de eficiencia de entrenamiento
  independiente del hardware, y complementa los tiempos absolutos del
  benchmark principal.

Protocolo
---------
  - Eb/N0 fijo en dos puntos representativos: 6 dB y 10 dB.
  - Barrido de N_train: [500, 1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k].
  - Para SVM RBF: N_train máximo limitado a 20k (escala cuadrática).
  - Test set: fijo, grande (200k símbolos piloto equiprobables) para todos los puntos.
  - Entrenamiento: secuencia piloto sintética (PRBS, equiprobable).
  - Hiperparámetros: cargados del caché del benchmark principal.
  - Se repite cada punto 3 veces (semillas distintas) → mediana para robustez.

Métricas reportadas
-------------------
  - BER absoluta (escala log)
  - BER normalizada = BER_clf / BER_teo  (1.0 = óptimo de Bayes)

Outputs
-------
  - resultados/curvas_aprendizaje.json     (datos para reproducir gráficas)
  - resultados/curvas_aprendizaje_6dB.png
  - resultados/curvas_aprendizaje_10dB.png
  - resultados/curvas_aprendizaje_normalizada.png  (ambos Eb/N0 en panel único)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── Path al raíz del proyecto ────────────────────────────────────────────────
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR_BASE)

import config
from canal.canal_awgn      import calcular_sigma, agregar_ruido, ber_teorica_16qam
from canal.modulador       import modular, simbolos_a_etiquetas
from metricas.metricas     import calcular_ber
from scrambler             import generar_prbs

from clasificadores.bayes          import ClasificadorBayes
from clasificadores.logistic_regression import ClasificadorLogisticRegression
from clasificadores.svm            import ClasificadorSVM_RBF, ClasificadorSVM_Lineal
from clasificadores.knn            import ClasificadorKNN
from clasificadores.random_forest  import ClasificadorRandomForest
from clasificadores.redes_neuronales import ClasificadorMLP, ClasificadorRedProfunda

# ── Parámetros del experimento ───────────────────────────────────────────────

EB_N0_PUNTOS  = [6.0, 10.0]          # dB

N_TRAIN_SWEEP = [500, 1_000, 2_000, 5_000, 10_000,
                 20_000, 50_000, 100_000, 200_000]

N_TRAIN_MAX_SVM_RBF = 20_000         # límite para SVM RBF (escala cuadrática)

N_TEST        = 200_000              # fijo para todos los puntos
N_REPETICIONES = 3                   # semillas distintas por punto → mediana

A = 1.0                              # amplitud de constelación

DIR_RESULTADOS = os.path.join(DIR_BASE, "resultados")
ARCHIVO_JSON   = os.path.join(DIR_RESULTADOS, "curvas_aprendizaje.json")

# ── Colores consistentes con el benchmark principal ──────────────────────────
COLORES = {
    "Bayes (ML)"   : "#000000",
    "SVM RBF"      : "#E53935",
    "SVM Lineal"   : "#FF7043",
    "KNN (k=51)"   : "#8E24AA",
    "Random Forest": "#1E88E5",
    "MLP Simple"   : "#43A047",
    "Red Profunda" : "#00ACC1",
    "Logistic Reg.": "#F57C00",
}
MARKERS = {
    "Bayes (ML)"   : "D",
    "SVM RBF"      : "s",
    "SVM Lineal"   : "^",
    "KNN (k=51)"   : "v",
    "Random Forest": "P",
    "MLP Simple"   : "o",
    "Red Profunda" : "X",
    "Logistic Reg.": "h",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Generación de datos
# ═══════════════════════════════════════════════════════════════════════════════

def generar_bits_piloto(n_simbolos: int, seed: int) -> np.ndarray:
    """Genera bitstream equiprobable de longitud n_simbolos*4 bits."""
    n_bits = n_simbolos * 4
    return generar_prbs(n_bits, seed=seed, poly=config.SCRAMBLING_POLY)


def construir_clasificadores() -> list:
    """
    Instancia todos los clasificadores con los hiperparámetros del caché.
    Bayes no tiene hiperparámetros. Los demás cargan del caché del benchmark.
    """
    clfs = [
        ClasificadorBayes(),

        ClasificadorSVM_RBF(
            C=config.SVM_C_DEFAULT,
            C_grid=config.SVM_C_GRID,
            optimizar=True,
            guardar_modelo=False,
            dir_modelos=config.DIR_MODELOS,
            usar_cache=True,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorSVM_Lineal(
            C=config.SVM_C_DEFAULT,
            C_grid=config.SVM_C_GRID,
            optimizar=True,
            guardar_modelo=False,
            dir_modelos=config.DIR_MODELOS,
            usar_cache=True,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorKNN(
            k=config.KNN_K_DEFAULT,
            k_grid=config.KNN_K_GRID,
            optimizar=True,
            guardar_modelo=False,
            dir_modelos=config.DIR_MODELOS,
            usar_cache=True,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorRandomForest(
            n_estimators=config.RF_N_TREES_DEFAULT,
            max_depth=config.RF_MAX_DEPTH_DEFAULT,
            n_trees_grid=config.RF_N_TREES_GRID,
            max_depth_grid=config.RF_MAX_DEPTH_GRID,
            optimizar=True,
            guardar_modelo=False,
            dir_modelos=config.DIR_MODELOS,
            seed=config.SEED,
            usar_cache=True,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorMLP(
            capas_ocultas=config.NN_MLP_HIDDEN,
            epochs=config.NN_MLP_EPOCHS,
            batch_size=config.NN_MLP_BATCH_SIZE,
            lr=config.NN_MLP_LR,
            lr_patience=config.NN_MLP_LR_PATIENCE,
            lr_factor=config.NN_MLP_LR_FACTOR,
            lr_min=config.NN_MLP_LR_MIN,
            guardar_modelo=False,
            dir_modelos=config.DIR_MODELOS,
            seed=config.SEED,
        ),

        ClasificadorRedProfunda(
            capas_ocultas=config.NN_DEEP_HIDDEN,
            epochs=config.NN_DEEP_EPOCHS,
            batch_size=config.NN_DEEP_BATCH_SIZE,
            lr=config.NN_DEEP_LR,
            lr_patience=config.NN_DEEP_LR_PATIENCE,
            lr_factor=config.NN_DEEP_LR_FACTOR,
            lr_min=config.NN_DEEP_LR_MIN,
            guardar_modelo=False,
            dir_modelos=config.DIR_MODELOS,
            seed=config.SEED,
        ),

        ClasificadorLogisticRegression(
            C=config.LR_C_DEFAULT,
            C_grid=config.LR_C_GRID,
            cv_folds=config.LR_CV_FOLDS,
            max_iter=config.LR_MAX_ITER,
            optimizar=True,
            usar_cache=True,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),
    ]
    return clfs


def evaluar_punto(clf, Eb_N0_dB: float, n_train: int,
                  seed_train: int, seed_test: int, seed_ruido: int) -> dict:
    """
    Entrena el clasificador con n_train símbolos y evalúa sobre N_TEST fijos.
    Devuelve BER y tiempo de entrenamiento.
    """
    sigma = calcular_sigma(Eb_N0_dB, A)
    rng   = np.random.default_rng(seed_ruido)

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    bits_train    = generar_bits_piloto(n_train, seed=seed_train)
    simb_tx_train = modular(bits_train, A=A)
    etiq_train    = simbolos_a_etiquetas(simb_tx_train, A=A)
    simb_rx_train = agregar_ruido(simb_tx_train, sigma, rng)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf._fit_interno(simb_rx_train, etiq_train)
    t_train = time.perf_counter() - t0
    clf._entrenado = True

    # ── Test (fijo, grande) ───────────────────────────────────────────────────
    bits_test    = generar_bits_piloto(N_TEST, seed=seed_test)
    simb_tx_test = modular(bits_test, A=A)
    etiq_test    = simbolos_a_etiquetas(simb_tx_test, A=A)
    simb_rx_test = agregar_ruido(simb_tx_test, sigma,
                                  np.random.default_rng(seed_ruido + 1000))

    etiq_pred = clf._predict_interno(simb_rx_test)

    # calcular_ber espera bits de referencia, no etiquetas
    from canal.modulador import etiquetas_a_bits
    bits_ref  = etiquetas_a_bits(etiq_test)
    ber, n_errores = calcular_ber(bits_ref, etiq_pred)

    return {
        "ber"      : float(ber),
        "n_errores": int(n_errores),
        "t_train"  : float(t_train),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Bucle principal
# ═══════════════════════════════════════════════════════════════════════════════

def correr_experimento() -> dict:
    """
    Ejecuta el experimento completo.
    Devuelve diccionario con resultados y lo guarda en JSON.
    """
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    # Cargar resultados previos si existen (para corridas parciales)
    if os.path.exists(ARCHIVO_JSON):
        with open(ARCHIVO_JSON) as f:
            resultados = json.load(f)
        print(f"  [Cache] Resultados previos cargados desde '{ARCHIVO_JSON}'")
        print(f"          Clasificadores ya procesados: "
              f"{list(resultados.get('datos', {}).keys())}")
    else:
        resultados = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "datos": {}}

    clfs = construir_clasificadores()

    # Hiperparámetros: cargar del caché (solo necesitamos correr optimizar
    # una vez con un set pequeño para que lean el JSON)
    print("\n  Cargando hiperparámetros del caché...")
    _sigma_dummy = calcular_sigma(8.0, A)
    _rng_dummy   = np.random.default_rng(0)
    _bits_dummy  = generar_bits_piloto(1000, seed=1)
    _simb_dummy  = modular(_bits_dummy, A=A)
    _simb_rx_dum = agregar_ruido(_simb_dummy, _sigma_dummy, _rng_dummy)
    _etiq_dummy  = simbolos_a_etiquetas(_simb_dummy, A=A)

    for clf in clfs:
        if not isinstance(clf, ClasificadorBayes):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.optimizar_hiperparametros(_simb_rx_dum, _etiq_dummy)

    n_total = (len(clfs) * len(EB_N0_PUNTOS) *
               len(N_TRAIN_SWEEP) * N_REPETICIONES)
    contador = 0

    for clf in clfs:
        nombre = clf.nombre

        # Límite de N_train para SVM RBF
        if "RBF" in nombre:
            sweep = [n for n in N_TRAIN_SWEEP if n <= N_TRAIN_MAX_SVM_RBF]
            print(f"\n  ⚠  {nombre}: N_train limitado a {N_TRAIN_MAX_SVM_RBF:,} "
                  f"(escala cuadrática)")
        else:
            sweep = N_TRAIN_SWEEP

        if nombre not in resultados["datos"]:
            resultados["datos"][nombre] = {}

        for Eb_N0_dB in EB_N0_PUNTOS:
            clave_eb = f"{Eb_N0_dB}dB"

            if clave_eb not in resultados["datos"][nombre]:
                resultados["datos"][nombre][clave_eb] = {}

            print(f"\n{'='*60}")
            print(f"  {nombre}  |  Eb/N0 = {Eb_N0_dB} dB")
            print(f"{'='*60}")

            for n_train in sweep:
                clave_n = str(n_train)

                # Saltar si ya está calculado
                if clave_n in resultados["datos"][nombre][clave_eb]:
                    print(f"  N={n_train:>7,}  [ya calculado, saltando]")
                    contador += N_REPETICIONES
                    continue

                bers      = []
                n_errores_lista = []
                t_trains  = []

                for rep in range(N_REPETICIONES):
                    seed_train = config.SEED + rep * 1000 + 1
                    seed_test  = config.SEED + rep * 1000 + 501
                    seed_ruido = config.SEED + rep * 1000 + 251

                    res = evaluar_punto(clf, Eb_N0_dB, n_train,
                                        seed_train, seed_test, seed_ruido)
                    bers.append(res["ber"])
                    n_errores_lista.append(res["n_errores"])
                    t_trains.append(res["t_train"])
                    contador += 1

                    progreso = contador / n_total * 100
                    print(f"  N={n_train:>7,}  rep {rep+1}/{N_REPETICIONES}"
                          f"  BER={res['ber']:.3e}"
                          f"  err={res['n_errores']:4d}"
                          f"  t={res['t_train']:6.1f}s"
                          f"  [{progreso:.1f}%]")

                # Guardar estadísticas (mediana + rango intercuartil)
                resultados["datos"][nombre][clave_eb][clave_n] = {
                    "ber_mediana"  : float(np.median(bers)),
                    "ber_p25"      : float(np.percentile(bers, 25)),
                    "ber_p75"      : float(np.percentile(bers, 75)),
                    "ber_vals"     : [float(b) for b in bers],
                    "n_errores"    : [int(e) for e in n_errores_lista],
                    "t_train_med"  : float(np.median(t_trains)),
                    "t_train_vals" : [float(t) for t in t_trains],
                }

                # Guardar progreso después de cada N_train completado
                resultados["fecha_ultima_actualizacion"] = \
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(ARCHIVO_JSON, "w") as f:
                    json.dump(resultados, f, indent=2)

    print(f"\n  ✓ Experimento completado. Datos guardados en '{ARCHIVO_JSON}'")
    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# Gráficas
# ═══════════════════════════════════════════════════════════════════════════════

def _color(nombre):
    for k, v in COLORES.items():
        if k in nombre or nombre in k:
            return v
    return "#888888"

def _marker(nombre):
    for k, v in MARKERS.items():
        if k in nombre or nombre in k:
            return v
    return "o"

def _guardar_fig(fig, nombre_archivo):
    ruta = os.path.join(DIR_RESULTADOS, nombre_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"  [Graficas] Guardado: '{ruta}'")


def graficar_curvas_ber(resultados: dict):
    """
    Figura A: BER vs N_train para cada Eb/N0, en escala log-log.
    Un panel por cada Eb/N0.
    """
    ber_teos = {str(eb): float(ber_teorica_16qam([eb])[0])
                for eb in EB_N0_PUNTOS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Curvas de aprendizaje — BER vs. N_train\n"
                 "16-QAM | Canal AWGN | Protocolo piloto equiprobable",
                 fontsize=13, fontweight='bold')

    for ax, Eb_N0_dB in zip(axes, EB_N0_PUNTOS):
        clave_eb  = f"{Eb_N0_dB}dB"
        ber_teo   = ber_teos[str(Eb_N0_dB)]

        # Línea de referencia teórica
        ax.axhline(ber_teo, color='black', linewidth=2.0,
                   linestyle='--', label=f'Bayes teórico ({ber_teo:.2e})',
                   zorder=10)

        for nombre, datos_clf in resultados["datos"].items():
            if clave_eb not in datos_clf:
                continue
            datos_eb = datos_clf[clave_eb]

            ns   = sorted([int(k) for k in datos_eb.keys()])
            meds = [datos_eb[str(n)]["ber_mediana"] for n in ns]
            p25  = [datos_eb[str(n)]["ber_p25"]     for n in ns]
            p75  = [datos_eb[str(n)]["ber_p75"]     for n in ns]

            color  = _color(nombre)
            marker = _marker(nombre)

            ax.plot(ns, meds, color=color, marker=marker,
                    linewidth=1.8, markersize=7,
                    label=nombre, zorder=5)

            # Banda de incertidumbre (percentiles 25-75)
            ax.fill_between(ns, p25, p75, color=color, alpha=0.12)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N_train (símbolos de trama piloto)', fontsize=12)
        ax.set_ylabel('BER', fontsize=12)
        ax.set_title(f'Eb/N0 = {Eb_N0_dB} dB', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.set_xlim(min(N_TRAIN_SWEEP) * 0.7, max(N_TRAIN_SWEEP) * 1.5)

    plt.tight_layout()
    _guardar_fig(fig, "curvas_aprendizaje_ber.png")
    plt.close(fig)


def graficar_curvas_normalizadas(resultados: dict):
    """
    Figura B: BER_clf / BER_teo vs N_train.
    Referencia horizontal en 1.0 = óptimo de Bayes.
    Hace comparables los dos Eb/N0 en un mismo panel.
    """
    ber_teos = {str(eb): float(ber_teorica_16qam([eb])[0])
                for eb in EB_N0_PUNTOS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Curvas de aprendizaje — BER normalizada respecto al óptimo de Bayes\n"
                 "Valor 1.0 = rendimiento del detector Bayesiano teórico",
                 fontsize=13, fontweight='bold')

    for ax, Eb_N0_dB in zip(axes, EB_N0_PUNTOS):
        clave_eb = f"{Eb_N0_dB}dB"
        ber_teo  = ber_teos[str(Eb_N0_dB)]

        # Referencia en 1.0
        ax.axhline(1.0, color='black', linewidth=2.0, linestyle='--',
                   label='Bayes teórico (1.00)', zorder=10)
        # Zona de "convergencia práctica": BER < 1.1 × BER_teo
        ax.axhline(1.1, color='gray', linewidth=1.0, linestyle=':',
                   label='Umbral convergencia (+10%)', zorder=9)

        for nombre, datos_clf in resultados["datos"].items():
            if clave_eb not in datos_clf:
                continue
            datos_eb = datos_clf[clave_eb]

            ns       = sorted([int(k) for k in datos_eb.keys()])
            meds_n   = [datos_eb[str(n)]["ber_mediana"] / ber_teo for n in ns]
            p25_n    = [datos_eb[str(n)]["ber_p25"]     / ber_teo for n in ns]
            p75_n    = [datos_eb[str(n)]["ber_p75"]     / ber_teo for n in ns]

            color  = _color(nombre)
            marker = _marker(nombre)

            ax.plot(ns, meds_n, color=color, marker=marker,
                    linewidth=1.8, markersize=7,
                    label=nombre, zorder=5)
            ax.fill_between(ns, p25_n, p75_n, color=color, alpha=0.12)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N_train (símbolos de trama piloto)', fontsize=12)
        ax.set_ylabel('BER_clf / BER_teórica', fontsize=12)
        ax.set_title(f'Eb/N0 = {Eb_N0_dB} dB', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.set_xlim(min(N_TRAIN_SWEEP) * 0.7, max(N_TRAIN_SWEEP) * 1.5)

    plt.tight_layout()
    _guardar_fig(fig, "curvas_aprendizaje_normalizada.png")
    plt.close(fig)


def graficar_n_convergencia(resultados: dict):
    """
    Figura C: tabla visual del N_min de convergencia por clasificador y Eb/N0.
    Convergencia: BER_clf < 1.10 × BER_teo (dentro del 10% del óptimo).
    """
    ber_teos = {str(eb): float(ber_teorica_16qam([eb])[0])
                for eb in EB_N0_PUNTOS}
    UMBRAL = 1.10

    nombres = list(resultados["datos"].keys())
    fig, ax = plt.subplots(figsize=(10, max(4, len(nombres) * 0.7 + 1.5)))
    fig.suptitle("N mínimo de convergencia por clasificador\n"
                 f"Criterio: BER_clf < {UMBRAL:.0%} × BER_teórica",
                 fontsize=13, fontweight='bold')

    x        = np.arange(len(EB_N0_PUNTOS))
    ancho    = 0.8 / len(nombres)
    offsets  = np.linspace(-(len(nombres)-1)/2, (len(nombres)-1)/2, len(nombres)) * ancho

    for i, nombre in enumerate(nombres):
        n_convs = []
        for Eb_N0_dB in EB_N0_PUNTOS:
            clave_eb = f"{Eb_N0_dB}dB"
            ber_teo  = ber_teos[str(Eb_N0_dB)]
            datos_eb = resultados["datos"][nombre].get(clave_eb, {})
            ns       = sorted([int(k) for k in datos_eb.keys()])

            n_conv = None
            for n in ns:
                ber_med = datos_eb[str(n)]["ber_mediana"]
                if ber_med / ber_teo <= UMBRAL:
                    n_conv = n
                    break

            n_convs.append(n_conv if n_conv else max(N_TRAIN_SWEEP) * 2)

        bars = ax.bar(x + offsets[i], n_convs,
                      width=ancho * 0.9,
                      color=_color(nombre),
                      label=nombre, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

        # Anotar el valor encima de cada barra
        for rect, val in zip(bars, n_convs):
            etiqueta = f"{val:,}" if val <= max(N_TRAIN_SWEEP) else ">200k"
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() * 1.05,
                    etiqueta, ha='center', va='bottom',
                    fontsize=7.5, rotation=45)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{eb} dB' for eb in EB_N0_PUNTOS], fontsize=12)
    ax.set_xlabel('Eb/N0', fontsize=12)
    ax.set_ylabel('N mínimo de trama piloto (símbolos)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    _guardar_fig(fig, "curvas_aprendizaje_n_convergencia.png")
    plt.close(fig)


def graficar_t_train_vs_n(resultados: dict):
    """
    Figura D: Tiempo de entrenamiento vs N_train (escala log-log).
    Permite ver la pendiente de cada técnica (complejidad empírica).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Tiempo de entrenamiento vs. N_train\n"
                 "Pendiente en log-log ≈ orden de complejidad empírico",
                 fontsize=13, fontweight='bold')

    for ax, Eb_N0_dB in zip(axes, EB_N0_PUNTOS):
        clave_eb = f"{Eb_N0_dB}dB"

        for nombre, datos_clf in resultados["datos"].items():
            if clave_eb not in datos_clf:
                continue
            datos_eb = datos_clf[clave_eb]
            ns    = sorted([int(k) for k in datos_eb.keys()])
            ts    = [datos_eb[str(n)]["t_train_med"] for n in ns]

            ax.plot(ns, ts, color=_color(nombre), marker=_marker(nombre),
                    linewidth=1.8, markersize=7, label=nombre)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N_train (símbolos)', fontsize=12)
        ax.set_ylabel('Tiempo de entrenamiento (s)', fontsize=12)
        ax.set_title(f'Eb/N0 = {Eb_N0_dB} dB', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        # Referencia de pendiente: O(N), O(N²)
        ns_ref = np.array([1e3, 2e5])
        t_ref  = ts[0] if ts else 1.0
        n_ref  = ns[0] if ns else 1000
        ax.plot(ns_ref, t_ref * (ns_ref/n_ref)**1,
                color='gray', linestyle=':', linewidth=1.0, label='O(N)')
        ax.plot(ns_ref, t_ref * (ns_ref/n_ref)**2,
                color='gray', linestyle='--', linewidth=1.0, label='O(N²)')

    plt.tight_layout()
    _guardar_fig(fig, "curvas_aprendizaje_tiempo.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("█" * 60)
    print("  CURVAS DE APRENDIZAJE — BER vs. N_train")
    print("  16-QAM | Canal AWGN | Protocolo piloto")
    print("█" * 60)
    print(f"\n  Eb/N0 evaluados : {EB_N0_PUNTOS} dB")
    print(f"  N_train barrido : {[f'{n:,}' for n in N_TRAIN_SWEEP]}")
    print(f"  N_test fijo     : {N_TEST:,} símbolos")
    print(f"  Repeticiones    : {N_REPETICIONES} por punto")
    print(f"  SVM RBF límite  : {N_TRAIN_MAX_SVM_RBF:,} símbolos")

    # Estimación de tiempo (muy aproximada)
    n_puntos_totales = (
        len([n for n in N_TRAIN_SWEEP if n <= N_TRAIN_MAX_SVM_RBF]) +  # SVM RBF
        len(N_TRAIN_SWEEP) * 6  # resto de clasificadores
    ) * len(EB_N0_PUNTOS) * N_REPETICIONES
    print(f"\n  Puntos totales a evaluar: {n_puntos_totales}")
    print(f"  Nota: SVM RBF a N=20k puede tardar ~15-30 min por punto.")
    print(f"  Los resultados se guardan incrementalmente — se puede interrumpir")
    print(f"  y retomar en cualquier momento.\n")

    input("  Presione ENTER para comenzar (o Ctrl+C para cancelar)...")

    resultados = correr_experimento()

    print("\n" + "="*60)
    print("  GENERANDO GRÁFICAS")
    print("="*60)
    graficar_curvas_ber(resultados)
    graficar_curvas_normalizadas(resultados)
    graficar_n_convergencia(resultados)
    graficar_t_train_vs_n(resultados)

    print("\n" + "█"*60)
    print("  EXPERIMENTO COMPLETADO")
    print(f"  Figuras guardadas en: {DIR_RESULTADOS}")
    print("█"*60)


if __name__ == "__main__":
    main()
