"""
main.py
=======
Orquestador principal del benchmark unificado.

Flujo completo:
  PASO 0 — Procesamiento de la fuente de información (audio → bitstream)
  PASO 1 — Validación del simulador (Bayes simulado vs. curva teórica)
  PASO 2 — Benchmark ML (barrido Eb/N0, entrenamiento y evaluación)
  PASO 3 — Generación de gráficas y tabla comparativa

Para modificar cualquier parámetro del experimento, editar config.py únicamente.
"""

import argparse
import numpy as np
import os
import sys
import time

# ---------------------------------------------------------------------------
# Configuración del path — permite ejecutar main.py desde cualquier ubicación
# ---------------------------------------------------------------------------
_DIR_PROYECTO = os.path.dirname(os.path.abspath(__file__))
if _DIR_PROYECTO not in sys.path:
    sys.path.insert(0, _DIR_PROYECTO)

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
import config

from fuente.fuente        import procesar_fuente, extraer_segmento
from canal.modulador      import (modular, simbolos_a_etiquetas,
                                  calcular_amplitud, verificar_equiprobabilidad_simbolos)
from canal.canal_awgn     import transmitir, ber_teorica_16qam
from validacion.validacion import validar_simulador

from clasificadores.bayes          import ClasificadorBayes
from clasificadores.svm            import ClasificadorSVM_RBF, ClasificadorSVM_Lineal
from clasificadores.knn            import ClasificadorKNN
from clasificadores.random_forest  import ClasificadorRandomForest
from clasificadores.redes_neuronales import ClasificadorMLP, ClasificadorRedProfunda
from clasificadores.elm_clf             import ClasificadorELM
from clasificadores.sgd_nystroem        import ClasificadorSGDNystroem
from clasificadores.logistic_regression import ClasificadorLogisticRegression

from metricas.metricas import (nuevo_registro_clasificador, registrar_punto,
                                imprimir_resumen, construir_tabla_comparativa,
                                imprimir_advertencias_estadisticas)
import hiperparametros_cache as cache
import resultados_cache as rcache
from graficas.graficas import (curva_ber_principal, curva_ber_zoom,
                                grafica_degradacion, tabla_metricas,
                                regiones_decision, historial_loss,
                                histograma_simbolos)


# ---------------------------------------------------------------------------
# Generador de secuencia piloto (entrenamiento)
# ---------------------------------------------------------------------------
def generar_bits_piloto(n_simbolos: int, seed: int = 2025) -> np.ndarray:
    """
    Genera una secuencia de bits pseudoaleatoria equiprobable para usar
    como secuencia piloto de entrenamiento.

    Emula el protocolo real: el transmisor envía esta secuencia conocida
    antes de los datos de usuario. El receptor la recibe con las
    características del canal embebidas y entrena el clasificador ML.

    A diferencia de los bits de voz (P(1) ≈ 0.58), esta secuencia garantiza
    P(1) = P(0) = 0.5 por construcción, cumpliendo el supuesto de equiprobabilidad
    del detector Bayesiano y ofreciendo condiciones justas para todos los clasificadores.

    Parameters
    ----------
    n_simbolos : int   número de símbolos 16-QAM (se generan n_simbolos * 4 bits)
    seed       : int   semilla (debe ser la misma en TX y RX para que ambos
                       conozcan la secuencia → protocolo piloto)

    Returns
    -------
    np.ndarray int8, shape (n_simbolos * 4,)
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_simbolos * 4, dtype=np.int8)


# ---------------------------------------------------------------------------
# Utilidad: separador visual
# ---------------------------------------------------------------------------
def _separador(titulo: str = ""):
    largo = 60
    print("\n" + "=" * largo)
    if titulo:
        pad = (largo - len(titulo) - 2) // 2
        print(" " * pad + f" {titulo} ")
        print("=" * largo)


# ---------------------------------------------------------------------------
# PASO 0: Fuente de información
# ---------------------------------------------------------------------------
def paso_fuente() -> dict:
    _separador("PASO 0 — FUENTE DE INFORMACIÓN")
    fuente = procesar_fuente(
        ruta_audio  = config.RUTA_AUDIO,
        mu          = config.MU,
        bits_cuant  = config.BITS_CUANT,
        fs_objetivo = config.FS_RESAMPLE,
        seed        = config.SEED,
    )

    # Verificar que hay suficientes símbolos para entrenamiento + test en todos los puntos
    simbolos_necesarios = (config.N_TRAIN + config.N_TEST) * len(config.EB_N0_RANGE_DB)
    if fuente['n_simbolos'] < config.N_TRAIN + config.N_TEST:
        print(f"\n⚠  ADVERTENCIA: La fuente tiene {fuente['n_simbolos']} símbolos.")
        print(f"   Se necesitan al menos {config.N_TRAIN + config.N_TEST} por punto.")
        print(f"   El mismo segmento se reutilizará entre puntos Eb/N0.")
        print(f"   Para resultados más robustos, usar un audio más largo.")

    return fuente


# ---------------------------------------------------------------------------
# PASO 1: Validación del simulador
# ---------------------------------------------------------------------------
def paso_validacion(fuente: dict) -> bool:
    if not config.VALIDAR_SIMULADOR:
        print("\n[main] Validación desactivada (VALIDAR_SIMULADOR=False en config.py)")
        return True

    _, aprobado = validar_simulador(
        fuente_dict    = fuente,
        Eb_N0_range_dB = config.EB_N0_RANGE_DB,
        seed           = config.SEED,
        guardar_figura = config.GUARDAR_GRAFICAS,
        dir_resultados = config.DIR_RESULTADOS,
    )
    return aprobado


# ---------------------------------------------------------------------------
# Construcción de clasificadores
# ---------------------------------------------------------------------------
def construir_clasificadores() -> list:
    """Instancia todos los clasificadores con los parámetros de config.py."""
    optimizar = config.OPTIMIZAR_HIPERPARAMETROS

    clfs = [
        ClasificadorBayes(),

        ClasificadorSVM_RBF(
            C=config.SVM_C_DEFAULT, gamma=config.SVM_GAMMA_DEFAULT,
            C_grid=config.SVM_C_GRID, gamma_grid=config.SVM_GAMMA_GRID,
            cv_folds=config.SVM_CV_FOLDS,
            optimizar=optimizar,
            guardar_modelo=config.GUARDAR_MODELOS,
            dir_modelos=config.DIR_MODELOS,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorSVM_Lineal(
            C=config.SVM_C_DEFAULT,
            C_grid=config.SVM_C_GRID,
            cv_folds=config.SVM_CV_FOLDS,
            optimizar=optimizar,
            guardar_modelo=config.GUARDAR_MODELOS,
            dir_modelos=config.DIR_MODELOS,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorKNN(
            k=config.KNN_K_DEFAULT,
            k_grid=config.KNN_K_GRID,
            optimizar=optimizar,
            guardar_modelo=config.GUARDAR_MODELOS,
            dir_modelos=config.DIR_MODELOS,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorRandomForest(
            n_estimators=config.RF_N_TREES_DEFAULT,
            max_depth=config.RF_MAX_DEPTH_DEFAULT,
            n_trees_grid=config.RF_N_TREES_GRID,
            max_depth_grid=config.RF_MAX_DEPTH_GRID,
            optimizar=optimizar,
            guardar_modelo=config.GUARDAR_MODELOS,
            dir_modelos=config.DIR_MODELOS,
            seed=config.SEED,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
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
            guardar_modelo=config.GUARDAR_MODELOS,
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
            guardar_modelo=config.GUARDAR_MODELOS,
            dir_modelos=config.DIR_MODELOS,
            seed=config.SEED,
        ),

        ClasificadorSGDNystroem(
            n_components=config.SGD_N_COMPONENTS_DEFAULT,
            gamma=config.SGD_GAMMA_DEFAULT,
            n_comp_grid=config.SGD_N_COMPONENTS_GRID,
            gamma_grid=config.SGD_GAMMA_GRID,
            optimizar=optimizar,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorELM(
            hidden_units=config.ELM_HIDDEN_DEFAULT,
            C=config.ELM_C_DEFAULT,
            hidden_units_grid=config.ELM_HIDDEN_GRID,
            C_grid=config.ELM_C_GRID,
            activation_grid=config.ELM_ACTIVATION_GRID,
            optimizar=optimizar,
            seed=config.SEED,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),

        ClasificadorLogisticRegression(
            C=config.LR_C_DEFAULT,
            C_grid=config.LR_C_GRID,
            cv_folds=config.LR_CV_FOLDS,
            max_iter=config.LR_MAX_ITER,
            optimizar=optimizar,
            usar_cache=config.USAR_CACHE_HIPERPARAMETROS,
            dir_cache=config.DIR_HIPERPARAMETROS,
        ),
    ]
    return clfs




# ---------------------------------------------------------------------------
# Selección interactiva (o automática) de clasificadores a correr
# ---------------------------------------------------------------------------
def paso_seleccion_clasificadores(clasificadores: list, auto: bool = False, arg_clf: str = None) -> list:
    """
    Muestra los clasificadores disponibles y los resultados guardados,
    y le pregunta al usuario cuáles quiere correr en esta sesión.

    Los clasificadores NO seleccionados se conservan desde el cache de
    resultados anteriores y aparecen en las gráficas finales igual.
    """
    _separador("SELECCIÓN DE CLASIFICADORES")

    # Mostrar resumen de resultados guardados
    rcache.imprimir_resumen(config.DIR_RESULTADOS)

    print()
    print("  Clasificadores disponibles:")
    for i, clf in enumerate(clasificadores):
        tiene_resultado = rcache.existe(config.DIR_RESULTADOS) and             clf.nombre in rcache.cargar(config.DIR_RESULTADOS)
        estado = "✓ resultado guardado" if tiene_resultado else "  sin resultado previo"
        print(f"  [{i+1}] {clf.nombre:<22}  {estado}")

    print()
    print("  ¿Qué clasificadores desea correr en esta sesión?")
    print("  [A] Todos  (corrida completa)")
    print("  [S] Seleccionar manualmente")
    print()

    if auto:
        opcion = "A"
    elif arg_clf:
        if arg_clf.upper() == "A":
            opcion = "A"
        else:
            opcion = "S"
            entrada_auto = arg_clf
    else:
        while True:
            opcion = input("  Ingrese opción (A/S): ").strip().upper()
            if opcion in ("A", "S"):
                break
            print("  Ingrese 'A' o 'S'.")

    if opcion == "A":
        print(f"\n  Se correrán todos los clasificadores ({len(clasificadores)}).")
        return clasificadores

    # Selección manual
    print()
    print("  Ingrese los números separados por coma (ej: 5,6 para MLP y Red Profunda):")
    for i, clf in enumerate(clasificadores):
        print(f"    {i+1}. {clf.nombre}")
    print()

    while True:
        if auto or arg_clf:
            entrada = entrada_auto
        else:
            entrada = input("  Clasificadores a correr: ").strip()
            
        try:
            indices = [int(x.strip()) - 1 for x in entrada.split(",")]
            if all(0 <= idx < len(clasificadores) for idx in indices) and len(indices) > 0:
                break
            print(f"  Ingrese números entre 1 y {len(clasificadores)}.")
            if auto or arg_clf:
                sys.exit(1) # Cortar si el argumento automático es inválido
        except ValueError:
            print("  Formato inválido. Use números separados por coma.")
            if auto or arg_clf:
                sys.exit(1)

    seleccionados = [clasificadores[i] for i in indices]
    no_seleccionados = [clasificadores[i] for i in range(len(clasificadores))
                        if i not in indices]

    print(f"\n  Se correrán: {[c.nombre for c in seleccionados]}")
    if no_seleccionados:
        print(f"  Se conservarán del cache: {[c.nombre for c in no_seleccionados]}")
        # Verificar que los no seleccionados tienen resultados guardados
        sin_cache = [c.nombre for c in no_seleccionados
                     if not rcache.existe(config.DIR_RESULTADOS) or
                        c.nombre not in rcache.cargar(config.DIR_RESULTADOS)]
        if sin_cache:
            print(f"\n  ⚠  ADVERTENCIA: Los siguientes clasificadores no tienen")
            print(f"     resultados guardados y no aparecerán en las gráficas finales:")
            for nombre in sin_cache:
                print(f"     • {nombre}")
            print(f"     Para incluirlos, córralos al menos una vez.")

    return seleccionados


# ---------------------------------------------------------------------------
# Decisión interactiva (o automática) sobre hiperparámetros
# ---------------------------------------------------------------------------
def paso_decision_hiperparametros(clasificadores: list, auto: bool = False, arg_cache: str = None) -> None:
    """
    Muestra el estado del cache de hiperparámetros y le pregunta al usuario
    qué quiere hacer: usar los valores guardados o ejecutar una nueva búsqueda.

    La decisión se aplica clasificador por clasificador si hay mezcla
    (algunos con cache, otros sin cache), o globalmente si todos tienen o
    todos no tienen cache.
    """
    import json, os

    _separador("HIPERPARÁMETROS — DECISIÓN")

    dir_cache    = config.DIR_HIPERPARAMETROS
    con_cache    = []   # clasificadores que tienen cache guardado
    sin_cache    = []   # clasificadores que no tienen cache

    # Clasificar cuáles tienen cache disponible
    for clf in clasificadores:
        if not hasattr(clf, 'usar_cache'):
            continue   # Bayes y redes no tienen hiperparámetros que buscar
        nombre_cache = getattr(clf, 'nombre_base', clf.nombre)
        if cache.existe(nombre_cache, dir_cache):
            con_cache.append(clf)
        else:
            sin_cache.append(clf)

    # ── Caso 1: ninguno tiene cache ───────────────────────────────────────
    if not con_cache:
        print("\n  No se encontraron hiperparámetros guardados de corridas anteriores.")
        print("  Se ejecutará GridSearch para todos los clasificadores.")
        print()
        print("  ⚠  ADVERTENCIA: La búsqueda de hiperparámetros puede demorar")
        print("     varios minutos dependiendo del hardware.")
        print()
        if not auto:
            input("  Presione ENTER para continuar...")
        _aplicar_decision_cache(clasificadores, usar=False)
        return

    # ── Caso 2: todos tienen cache ────────────────────────────────────────
    if not sin_cache:
        print("\n  Se encontraron hiperparámetros guardados para TODOS los clasificadores:\n")
        _mostrar_detalle_cache(con_cache, dir_cache)

        print()
        print("  ¿Qué desea hacer?")
        print("  [1] Usar los hiperparámetros guardados  (recomendado, mucho más rápido)")
        print("  [2] Ejecutar nueva búsqueda GridSearch  (puede demorar varios minutos)")
        print()

        if auto:
            opcion = "1"
        elif arg_cache in ("1", "2"):
            opcion = arg_cache
        else:
            opcion = _pedir_opcion(["1", "2"])

        if opcion == "1":
            print("\n  ✓ Se usarán los hiperparámetros del cache.")
            _aplicar_decision_cache(clasificadores, usar=True)
        else:
            print("\n  ⚠  Se ejecutará una nueva búsqueda. Esto puede demorar.")
            print("     Los nuevos valores encontrados sobreescribirán el cache actual.")
            _aplicar_decision_cache(clasificadores, usar=False)
        return

    # ── Caso 3: mezcla — algunos con cache, otros sin cache ──────────────
    print(f"\n  Situación mixta:")
    print(f"  • {len(con_cache)} clasificador(es) CON cache guardado")
    print(f"  • {len(sin_cache)} clasificador(es) SIN cache (requieren búsqueda)\n")

    print("  Clasificadores CON cache:")
    _mostrar_detalle_cache(con_cache, dir_cache)

    print()
    print("  Clasificadores SIN cache (se ejecutará GridSearch obligatoriamente):")
    for clf in sin_cache:
        print(f"    • {clf.nombre}")

    print()
    print("  ¿Qué desea hacer con los clasificadores que SÍ tienen cache?")
    print("  [1] Usar los hiperparámetros guardados para esos clasificadores")
    print("  [2] Ejecutar nueva búsqueda para TODOS  (puede demorar varios minutos)")
    print()

    if auto:
        opcion = "1"
    elif arg_cache in ("1", "2"):
        opcion = arg_cache
    else:
        opcion = _pedir_opcion(["1", "2"])

    if opcion == "1":
        print("\n  ✓ Se usará el cache donde esté disponible.")
        print("    GridSearch solo para los clasificadores sin cache.")
        _aplicar_decision_cache(clasificadores, usar=True)
    else:
        print("\n  ⚠  Nueva búsqueda para todos. Esto puede demorar.")
        _aplicar_decision_cache(clasificadores, usar=False)


def _mostrar_detalle_cache(clasificadores_con_cache: list, dir_cache: str) -> None:
    """Muestra una tabla con el detalle de cada cache disponible."""
    import json, os

    print(f"  {'Clasificador':<20} {'Parámetros óptimos':<35} {'CV Accuracy':>12}  {'Calculado el'}")
    print("  " + "-"*85)

    for clf in clasificadores_con_cache:
        nombre_cache = getattr(clf, 'nombre_base', clf.nombre)
        nombre_archivo = nombre_cache.replace(" ", "_").lower() + "_hiperparams.json"
        ruta = os.path.join(dir_cache, nombre_archivo)

        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                datos = json.load(f)

            params  = datos.get("params", {})
            fecha   = datos.get("fecha", "desconocida")
            acc     = datos.get("cv_accuracy")
            acc_str = f"{acc:.4f}" if acc is not None else "  N/A"

            # Contexto adicional si está disponible
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())

            print(f"  {clf.nombre:<20} {params_str:<35} {acc_str:>12}  {fecha}")

        except Exception:
            print(f"  {clf.nombre:<20} {'(error al leer cache)':<35}")


def _pedir_opcion(opciones_validas: list) -> str:
    """Solicita al usuario una opción válida en bucle hasta obtenerla."""
    while True:
        resp = input(f"  Ingrese opción ({'/'.join(opciones_validas)}): ").strip()
        if resp in opciones_validas:
            return resp
        print(f"  Por favor ingrese una de las siguientes opciones: {opciones_validas}")


def _aplicar_decision_cache(clasificadores: list, usar: bool) -> None:
    """Aplica la decisión del usuario a todos los clasificadores que soporten cache."""
    for clf in clasificadores:
        if hasattr(clf, 'usar_cache'):
            clf.usar_cache = usar


# ---------------------------------------------------------------------------
# PASO 2: Benchmark ML
# ---------------------------------------------------------------------------
def paso_benchmark(fuente: dict, clasificadores: list) -> list:
    """
    Bucle principal del benchmark.
    Para cada Eb/N0:
      1. Genera los segmentos de entrenamiento y test desde la fuente.
      2. Modula, transmite por el canal AWGN.
      3. Entrena cada clasificador con los símbolos recibidos en entrenamiento.
      4. Evalúa en el conjunto de test.
      5. Registra todas las métricas.
    """
    _separador("PASO 2 — BENCHMARK ML")

    # ── Validar punto de optimización ────────────────────────────────────
    Eb_N0_opt = config.EB_N0_PARA_OPTIMIZACION_DB
    if Eb_N0_opt not in config.EB_N0_RANGE_DB:
        print(f"  ⚠  ADVERTENCIA: EB_N0_PARA_OPTIMIZACION_DB={Eb_N0_opt} dB no está en")
        print(f"     EB_N0_RANGE_DB={config.EB_N0_RANGE_DB}.")
        Eb_N0_opt = min(config.EB_N0_RANGE_DB, key=lambda x: abs(x - Eb_N0_opt))
        print(f"     → Ajustado al punto más cercano: {Eb_N0_opt} dB")

    print(f"\n  Punto de optimización de hiperparámetros: {Eb_N0_opt} dB")
    print(f"  Todos los puntos del barrido usarán estos hiperparámetros.")

    rng = np.random.default_rng(config.SEED)

    # Registros de métricas por clasificador
    registros = [nuevo_registro_clasificador(clf.nombre) for clf in clasificadores]

    # Cuántos símbolos hay en la fuente
    n_disponibles = fuente['n_simbolos']
    n_necesarios  = config.N_TRAIN + config.N_TEST

    if n_disponibles >= n_necesarios:
        inicio_train = 0
        inicio_test  = config.N_TRAIN
    else:
        print(f"\n  ⚠ Audio corto: reutilizando segmento. "
              f"Disponibles={n_disponibles}, Necesarios={n_necesarios}")
        inicio_train = 0
        inicio_test  = 0

    # ── FASE 0: Optimización de hiperparámetros ANTES del barrido ────────
    # Se transmite a través del canal al Eb/N0 de optimización, se corre el
    # GridSearch una única vez y los parámetros quedan fijos para todo el barrido.
    _separador(f"OPTIMIZACIÓN DE HIPERPARÁMETROS @ {Eb_N0_opt} dB")

    A = 1.0
    if config.MODO_ENTRENAMIENTO == 'piloto':
        n_train_opt    = config.N_TRAIN
        bits_train_opt = generar_bits_piloto(n_train_opt, seed=config.PILOTO_SEED)
    else:
        n_train_opt    = min(config.N_TRAIN, n_disponibles - inicio_train) if \
                             inicio_train < n_disponibles else n_disponibles
        bits_train_opt = extraer_segmento(fuente, inicio_train, n_train_opt,
                           usar_scrambling=config.USAR_SCRAMBLING,
                           scrambling_seed=config.SCRAMBLING_SEED,
                           scrambling_poly=config.SCRAMBLING_POLY)
    simbolos_tx_opt   = modular(bits_train_opt, A=A)
    etiquetas_tx_opt  = simbolos_a_etiquetas(simbolos_tx_opt, A=A)
    simbolos_rx_opt, sigma_opt = transmitir(simbolos_tx_opt, Eb_N0_opt, A, rng)

    print(f"  σ = {sigma_opt:.5f} | N_train = {n_train_opt} símbolos")

    hiperparams_globales = {}   # almacena los mejores params de cada clasificador
    for clf in clasificadores:
        if isinstance(clf, ClasificadorBayes):
            continue
        print(f"\n  → Optimizando: {clf.nombre}")
        params = clf.optimizar_hiperparametros(simbolos_rx_opt, etiquetas_tx_opt)
        hiperparams_globales[clf.nombre] = params

    print(f"\n  ✓ Optimización completada. Iniciando barrido con hiperparámetros fijos.")

    # ── Preparar segmento de test (fijo para todo el barrido) ──────────
    # El test usa SIEMPRE el bitstream de voz real, comenzando donde
    # termina el segmento de entrenamiento (si el modo es 'fuente'),
    # o desde el inicio (si el entrenamiento usa piloto sintético).
    # Esto garantiza que la evaluación siempre sea sobre datos reales.

    # ── FASE 1: Barrido completo de Eb/N0 ────────────────────────────────
    for i_eb, Eb_N0_dB in enumerate(config.EB_N0_RANGE_DB):
        _separador(f"Eb/N0 = {Eb_N0_dB} dB  ({i_eb+1}/{len(config.EB_N0_RANGE_DB)})")

        A = 1.0

        # ── Bits de entrenamiento ─────────────────────────────────────────
        # MODO 'piloto': secuencia pseudoaleatoria conocida, equiprobable.
        #   Emula el envío de una trama piloto antes de los datos reales.
        #   El clasificador se entrena sobre esta secuencia cuyas etiquetas
        #   son conocidas en el receptor (protocolo supervisado realista).
        # MODO 'fuente': segmento de la voz (comportamiento anterior).
        if config.MODO_ENTRENAMIENTO == 'piloto':
            bits_train = generar_bits_piloto(config.N_TRAIN,
                                             seed=config.PILOTO_SEED)
            n_train    = config.N_TRAIN
            origen_train = "piloto sintetico (equiprobable)"
        else:
            n_train    = min(config.N_TRAIN, n_disponibles - inicio_train) if \
                             inicio_train < n_disponibles else n_disponibles
            bits_train = extraer_segmento(fuente, inicio_train, n_train,
                           usar_scrambling=config.USAR_SCRAMBLING,
                           scrambling_seed=config.SCRAMBLING_SEED,
                           scrambling_poly=config.SCRAMBLING_POLY)
            origen_train = "voz (fuente real)"

        # ── Bits de test (N adaptativo para confiabilidad estadística) ───
        # MODO 'fuente': voz real — condición realista post-entrenamiento.
        # MODO 'piloto': secuencia sintética — aísla el efecto del canal.
        from canal.canal_awgn import ber_teorica_16qam as _ber_teo
        N_ERRORES_MIN_TEST = 100
        ber_ref      = _ber_teo([Eb_N0_dB])[0]
        n_test_ideal = int(np.ceil(N_ERRORES_MIN_TEST / (ber_ref * 4)))

        if config.MODO_TEST == 'fuente':
            n_test_max   = n_disponibles - inicio_test if n_disponibles > inicio_test \
                           else n_disponibles
            n_test       = min(max(config.N_TEST, n_test_ideal),
                               config.N_MAX_TEST, n_test_max)
            bits_test    = extraer_segmento(fuente, inicio_test, n_test,
                             usar_scrambling=config.USAR_SCRAMBLING,
                             scrambling_seed=config.SCRAMBLING_SEED,
                             scrambling_poly=config.SCRAMBLING_POLY)
            if config.USAR_SCRAMBLING:
                origen_test = "voz + scrambling (equiprobable)"
            else:
                origen_test = "voz sin scrambling (fuente real)"
        else:
            n_test       = min(max(config.N_TEST, n_test_ideal), config.N_MAX_TEST)
            bits_test    = generar_bits_piloto(n_test, seed=config.PILOTO_SEED + 1)
            origen_test  = "piloto sintetico (equiprobable)"

        es_confiable_esperado = (n_test >= n_test_ideal)

        # ── Modular ───────────────────────────────────────────────────────
        simbolos_tx_train = modular(bits_train, A=A)
        simbolos_tx_test  = modular(bits_test,  A=A)

        etiquetas_tx_train = simbolos_a_etiquetas(simbolos_tx_train, A=A)
        etiquetas_tx_test  = simbolos_a_etiquetas(simbolos_tx_test,  A=A)

        # ── Canal AWGN ────────────────────────────────────────────────────
        simbolos_rx_train, sigma = transmitir(simbolos_tx_train, Eb_N0_dB, A, rng)
        simbolos_rx_test,  _     = transmitir(simbolos_tx_test,  Eb_N0_dB, A, rng)

        conf_str = "confiable" if es_confiable_esperado else \
                   f"INSUFICIENTE (necesita {n_test_ideal:,})"
        print(f"\n  sigma = {sigma:.5f}")
        print(f"  TRAIN: N = {n_train:,}  [{origen_train}]")
        print(f"  TEST:  N = {n_test:,}  [{origen_test}]  [{conf_str}]")

        verificar_equiprobabilidad_simbolos(etiquetas_tx_train)

        # ── Entrenar y evaluar cada clasificador ──────────────────────────
        for clf, reg in zip(clasificadores, registros):
            print(f"\n  -> {clf.nombre}")

            if hasattr(clf, 'set_amplitud'):
                clf.set_amplitud(A)

            mejores_params = hiperparams_globales.get(clf.nombre, {})

            clf.fit(simbolos_rx_train, etiquetas_tx_train)
            print(f"    t_entrenamiento = {clf.tiempo_entrenamiento:.3f} s")

            etiquetas_det = clf.predict(simbolos_rx_test)
            print(f"    t_inferencia_1  = {clf.tiempo_inferencia_unitaria*1e6:.4f} us")

            registrar_punto(
                registro         = reg,
                Eb_N0_dB         = Eb_N0_dB,
                bits_ref         = bits_test,
                etiquetas_ref    = etiquetas_tx_test,
                etiquetas_det    = etiquetas_det,
                t_entrenamiento  = clf.tiempo_entrenamiento,
                t_inferencia_1   = clf.tiempo_inferencia_unitaria,
                mejores_params   = mejores_params,
                n_test_simbolos  = n_test,
                flops_inferencia = clf.flops_inferencia,
            )

            ber_actual  = reg['ber'][-1]
            acc_actual  = reg['accuracy'][-1]
            errores_act = reg['n_errores'][-1]
            conf_act    = "OK" if reg['confiable'][-1] else \
                          f"INSUFICIENTE ({errores_act} errores)"
            print(f"    BER = {ber_actual:.4e}  |  Accuracy = {acc_actual*100:.2f}%  |  {conf_act}")

    return registros


# ---------------------------------------------------------------------------
# PASO 3: Gráficas y resultados finales
# ---------------------------------------------------------------------------
def paso_graficas(registros: list, clasificadores: list, fuente: dict = None):
    _separador("PASO 3 — GRÁFICAS Y RESULTADOS")

    os.makedirs(config.DIR_RESULTADOS, exist_ok=True)

    # Resumen en consola
    for reg in registros:
        imprimir_resumen(reg)

    # Advertencias estadísticas consolidadas
    imprimir_advertencias_estadisticas(registros)

    # Tabla comparativa
    tabla = construir_tabla_comparativa(registros)
    print("\n  ── Tabla comparativa (promedios sobre todo el barrido) ──")
    print(f"  {'Clasificador':<20} {'BER media':>12} {'Accuracy':>10} "
          f"{'T.train(s)':>12} {'T.inf(µs)':>11} {'FLOPs/símbolo':>15}")
    print("  " + "-"*86)
    for nombre, metricas in tabla.items():
        ber_str   = f"{metricas['ber_media']:.3e}" if metricas['ber_media'] > 0 else "< 10⁻⁵"
        flops     = metricas.get('flops_inferencia', 0)
        flops_str = f"{flops:,}" if flops > 0 else "  N/A"
        print(
            f"  {nombre:<20} {ber_str:>12} "
            f"{metricas['accuracy_media']*100:>9.2f}% "
            f"{metricas['t_train_media_s']:>12.3f} "
            f"{metricas['t_inferencia_media_us']:>11.4f} "
            f"{flops_str:>15}"
        )

    # Gráficas
    guardar = config.GUARDAR_GRAFICAS
    dir_r   = config.DIR_RESULTADOS

    # Figura 0: Histograma de símbolos (análisis de fuente)
    histograma_simbolos(fuente,
                        usar_scrambling=config.USAR_SCRAMBLING,
                        scrambling_seed=config.SCRAMBLING_SEED,
                        scrambling_poly=config.SCRAMBLING_POLY,
                        guardar=guardar, dir_resultados=dir_r)

    # Figura 1: BER completa
    curva_ber_principal(registros, guardar=guardar, dir_resultados=dir_r)

    # Figura 2: BER zoom en región útil
    curva_ber_zoom(registros,      guardar=guardar, dir_resultados=dir_r)

    # Figura 3: Degradación en dB respecto a Bayes
    grafica_degradacion(registros, guardar=guardar, dir_resultados=dir_r)

    # Figura 4: Tabla comparativa de métricas
    tabla_metricas(registros,      guardar=guardar, dir_resultados=dir_r)

    # Regiones de decisión para el punto de Eb/N0 intermedio
    Eb_N0_regiones = config.EB_N0_RANGE_DB[len(config.EB_N0_RANGE_DB)//2]
    clfs_entrenados = [clf for clf in clasificadores if clf._entrenado]
    regiones_decision(clfs_entrenados, Eb_N0_dB=Eb_N0_regiones,
                      guardar=guardar, dir_resultados=dir_r)

    # Historial de loss de redes neuronales (último punto entrenado)
    redes = [clf for clf in clasificadores
             if hasattr(clf, '_historial_loss') and clf._historial_loss]
    if redes:
        historial_loss(redes, Eb_N0_dB=config.EB_N0_RANGE_DB[-1],
                       guardar=guardar, dir_resultados=dir_r)


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Unificado - Clasificadores ML para 16-QAM")
    parser.add_argument("--auto", action="store_true", help="Ejecutar en modo automático (corre todos los clasificadores, usa el caché por defecto).")
    parser.add_argument("--clasificadores", type=str, help="Clasificadores a correr. 'A' para todos, o lista de índices separados por coma (ej: '1,2').")
    parser.add_argument("--cache", type=str, choices=["1", "2"], help="Decisión de caché: '1' usar caché, '2' forzar nueva búsqueda GridSearch.")
    args = parser.parse_args()

    t_inicio_total = time.perf_counter()

    print("\n" + "█"*60)
    print("  BENCHMARK UNIFICADO — CLASIFICADORES ML PARA 16-QAM")
    print("  Canal AWGN | Protocolo de entrenamiento realista")
    print("█"*60)

    # PASO 0: Fuente
    fuente = paso_fuente()

    # PASO 1: Validación del simulador
    aprobado = paso_validacion(fuente)
    if not aprobado:
        print("\n  ✗ Benchmark detenido. Corregir la cadena de simulación.")
        sys.exit(1)

    # PASO 2: Decisión de hiperparámetros + Benchmark ML
    clasificadores = construir_clasificadores()
    clasificadores = paso_seleccion_clasificadores(clasificadores, auto=args.auto, arg_clf=args.clasificadores)
    paso_decision_hiperparametros(clasificadores, auto=args.auto, arg_cache=args.cache)
    registros      = paso_benchmark(fuente, clasificadores)

    # Fusionar con resultados anteriores y guardar
    registros = rcache.registros_completos(registros, config.DIR_RESULTADOS,
                                       seed=config.SEED)
    snap = {"Eb_N0_range_dB": config.EB_N0_RANGE_DB,
            "N_train": config.N_TRAIN, "N_test": config.N_TEST}
    rcache.guardar(registros, config.DIR_RESULTADOS, snap,
                   seed=config.SEED)

    # PASO 3: Gráficas
    paso_graficas(registros, clasificadores, fuente)

    t_total = time.perf_counter() - t_inicio_total
    _separador("BENCHMARK COMPLETADO")
    print(f"  Tiempo total de ejecución: {t_total/60:.1f} minutos")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
