"""
correr_lote.py
==============
Script orquestador para automatizar múltiples corridas del benchmark
con distintas semillas.

- Lee de config.py qué clasificadores deben correrse.
- Pregunta la cantidad de corridas deseadas.
- Propone semillas aleatorias automáticamente.
- Modifica config.py al vuelo para inyectar la semilla.
- Ejecuta main.py en modo automático.
- Renombra el archivo JSON resultante para que cada corrida quede separada.
"""

import sys
import os
import json
import random
import subprocess
import re
import time
import importlib

# Configurar UTF-8 en la salida estándar de Windows para evitar errores de codificación
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Importamos config para saber qué clasificadores vamos a correr
import config
import main as main_script

# ---------------------------------------------------------------------------
# SEMILLAS FIJAS
# ---------------------------------------------------------------------------
# Si la lista NO está vacía, el lote re-usa exactamente estas semillas (en este
# orden) en lugar de generarlas al azar, y NO pregunta la cantidad de corridas.
# Útil para reproducir un conjunto previo (p.ej. las 11 semillas del artículo)
# y comparar pre/post-fix de configuración punto a punto.
# Dejar como [] para volver al comportamiento aleatorio interactivo.
SEEDS_FIJAS = [58407, 82779, 76075, 27440, 61732, 85829,
               76296, 62105, 64731, 12343, 68651]

def update_config_seed(new_seed):
    """Actualiza el valor de SEED en config.py mediante expresiones regulares"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Busca la línea "SEED = <algo>" y la reemplaza
    content_new = re.sub(r"^SEED\s*=\s*\d+", f"SEED = {new_seed}", content, flags=re.MULTILINE)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content_new)

def seleccionar_clasificadores():
    """Pregunta de forma interactiva qué clasificadores correr para el lote."""
    todos = main_script.construir_clasificadores()
    
    print("\n  Clasificadores disponibles:")
    for i, clf in enumerate(todos):
        print(f"  [{i+1}] {clf.nombre}")
        
    print("\n  ¿Qué clasificadores deseas correr en todas las simulaciones de este lote?")
    print("  [A] Todos")
    print("  [S] Seleccionar manualmente (para omitir algunos)")
    print()
    
    while True:
        opcion = input("  Ingrese opción (A/S): ").strip().upper()
        if opcion in ("A", "S"):
            break
        print("  Ingrese 'A' o 'S'.")
        
    if opcion == "A":
        print("\n  ✓ Se correrán TODOS los clasificadores en este lote.")
        return "A"
        
    print("\n  Ingrese los números separados por coma que SÍ quiere correr (ej: 1,5,6):")
    while True:
        entrada = input("  Clasificadores a correr: ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in entrada.split(",")]
            if all(0 <= idx < len(todos) for idx in indices) and len(indices) > 0:
                print("\n  Clasificadores configurados para este lote:")
                for i, clf in enumerate(todos):
                    if i in indices:
                        print(f"    ✓ {clf.nombre}")
                    else:
                        print(f"    - {clf.nombre} (omitido)")
                return entrada
            print(f"  Ingrese números entre 1 y {len(todos)}.")
        except ValueError:
            print("  Formato inválido. Use números separados por coma.")

def preflight_gpu_check():
    """
    Detecta qué clasificadores podrán correr en GPU y cuáles no.
    Si hay una discrepancia (mezcla de GPU/CPU), avisa al usuario y le
    pregunta cómo proceder:
      [1] Todos en CPU  → modo seguro, comparable entre sí
      [2] GPU donde sea posible → máxima velocidad, pero los resultados
          de FLOPs/tiempo no serán directamente comparables entre modelos.

    Retorna:
      force_cpu (bool): True si el usuario eligió modo todo-CPU.
    """
    print("\n" + "-" * 60)
    print("  PRE-FLIGHT: Verificación de hardware")
    print("-" * 60)

    # --- 1. Verificar disponibilidad de GPU (CUDA/PyTorch) ---
    cuda_disponible = False
    gpu_nombre      = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            cuda_disponible = True
            gpu_nombre = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # --- 2. Verificar ThunderSVM (GPU-SVM) ---
    thundersvm_disponible = False
    try:
        import thundersvm  # noqa – sólo verificamos si importa sin error
        thundersvm_disponible = True
    except Exception:
        pass

    # --- 3. Determinar qué modelos correrán en GPU/CPU ---
    modelos_gpu  = []
    modelos_cpu  = []

    if cuda_disponible:
        modelos_gpu.append("Redes Neuronales (MLP, Red Profunda)")
        modelos_gpu.append("ELM (Extreme Learning Machine)")
        modelos_gpu.append("XGBoost")
    else:
        modelos_cpu.append("Redes Neuronales (MLP, Red Profunda)")
        modelos_cpu.append("ELM (Extreme Learning Machine)")
        modelos_cpu.append("XGBoost")

    # SVM necesita ThunderSVM compilado para GPU
    if cuda_disponible and thundersvm_disponible:
        modelos_gpu.append("SVM RBF / SVM Lineal")
    else:
        modelos_cpu.append("SVM RBF / SVM Lineal")

    # Estos modelos siempre corren en CPU
    modelos_cpu += ["Bayes", "KNN", "Random Forest", "Regresión Logística", "SGD-Nyström"]

    # --- 4. Mostrar resumen ---
    print()
    if cuda_disponible:
        print(f"  ✓ GPU detectada: {gpu_nombre}")
    else:
        print("  ✗ No se detectó GPU compatible con CUDA.")

    if modelos_gpu:
        print("\n  Modelos que pueden usar GPU:")
        for m in modelos_gpu:
            print(f"    ▶ {m}")

    # Modelos que forzosamente usan CPU (excluir los siempre-CPU conocidos)
    modelos_cpu_advertencia = [m for m in modelos_cpu
                                if m not in ("Bayes", "KNN", "Random Forest",
                                             "Regresión Logística", "SGD-Nyström")]
    if cuda_disponible and modelos_cpu_advertencia:
        print("\n  ⚠ ADVERTENCIA — Los siguientes modelos NO pueden usar GPU")
        print("    y correrán en CPU aunque se haya detectado una GPU:")
        for m in modelos_cpu_advertencia:
            print(f"    ✗ {m}")
        if not thundersvm_disponible:
            print("\n    → SVM: ThunderSVM no está compilado para este sistema.")
            print("      Para habilitarlo en GPU, compile ThunderSVM desde su código fuente.")

    # --- 5. Tomar decisión si hay mezcla GPU/CPU ---
    # Si TODOS los modelos van a CPU, no hay nada que preguntar.
    if not modelos_gpu or not cuda_disponible:
        print("\n  → Todos los modelos correrán en CPU.")
        return True  # force_cpu = True

    # Si hay modelos que no pueden ir a GPU, preguntar al usuario
    if modelos_cpu_advertencia:
        print()
        print("  " + "─" * 56)
        print("  IMPORTANTE: Para comparar el costo computacional (FLOPs,")
        print("  tiempos) entre modelos, TODOS deben correr en el MISMO")
        print("  dispositivo. Mezclar GPU y CPU hace la comparación inválida.")
        print("  " + "─" * 56)
        print()
        print("  ¿Cómo desea proceder?")
        print("  [1] Todos en CPU  (recomendado para comparar benchmarks)")
        print("  [2] GPU donde sea posible, CPU para el resto")
        print("       (más rápido, pero comparación de FLOPs/tiempos no válida)")
        print()
        while True:
            opcion = input("  Ingrese opción (1 / 2): ").strip()
            if opcion == "1":
                print("\n  ✓ Modo seleccionado: TODOS en CPU.")
                return True   # force_cpu
            elif opcion == "2":
                print("\n  ⚠ Modo seleccionado: GPU donde sea posible.")
                print("    (Los tiempos y FLOPs no serán comparables entre modelos.)")
                return False  # no force_cpu
            else:
                print("  Ingrese 1 o 2.")
    else:
        # Todos los modelos soportados van a GPU → no hay conflicto
        print("  ✓ Todos los modelos compatibles correrán en GPU.")
        return False  # no force_cpu


def update_config_force_cpu(force_cpu: bool):
    """Actualiza FORCE_CPU en config.py para que todos los sub-procesos lo hereden."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    val = "True" if force_cpu else "False"
    content_new = re.sub(r"^FORCE_CPU\s*=\s*\S+", f"FORCE_CPU = {val}",
                         content, flags=re.MULTILINE)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content_new)


def main():
    print("=" * 60)
    print("  ORQUESTADOR DE CORRIDAS MÚLTIPLES (BATCH RUNNER)")
    print("=" * 60)

    # Pre-flight: GPU check + decisión de hardware
    force_cpu = preflight_gpu_check()
    update_config_force_cpu(force_cpu)

    seleccion_str = seleccionar_clasificadores()

    if SEEDS_FIJAS:
        # Re-usar exactamente las semillas fijas configuradas arriba.
        seeds = list(SEEDS_FIJAS)
        n_corridas = len(seeds)
        print("-" * 60)
        print(f"\n✓ Usando {n_corridas} semillas FIJAS (reproducción):")
        print(f"  {seeds}")
    else:
        # 1. Solicitar cantidad de corridas
        try:
            print("-" * 60)
            entrada = input("\n¿Cuántas corridas independientes deseas ejecutar? ")
            n_corridas = int(entrada.strip())
            if n_corridas <= 0:
                raise ValueError
        except ValueError:
            print("Error: Ingresa un número entero positivo.")
            sys.exit(1)

        # 2. Generar semillas inteligentemente
        seeds = random.sample(range(10000, 99999), n_corridas)
        print(f"\n✓ Se han generado {n_corridas} semillas automáticamente:")
        print(f"  {seeds}")
    
    print("\nIniciando en 3 segundos (Presiona Ctrl+C para cancelar)...")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
        sys.exit(0)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "resultados")
    # Reanudar tras interrupción: omitir las semillas que YA están presentes
    # en el JSON único (resultados_benchmark.json), donde main.py acumula
    # todas las corridas del lote.
    json_unico = os.path.join(result_dir, "resultados_benchmark.json")
    seeds_existentes = set()
    if os.path.exists(json_unico):
        try:
            with open(json_unico, "r", encoding="utf-8") as f:
                datos_prev = json.load(f)
            seeds_existentes = {c.get("seed") for c in datos_prev.get("corridas", [])}
        except (json.JSONDecodeError, OSError):
            pass
    seeds = [s for s in seeds if s not in seeds_existentes]
    print(f"\n✓ {len(seeds)} semillas restantes tras filtrar las ya presentes "
          f"en resultados_benchmark.json.")
    
    # 3. Bucle principal de ejecución
    t_inicio_lote = time.perf_counter()
    
    for i, seed in enumerate(seeds):
        print("\n" + "█" * 60)
        print(f"  CORRIDA [{i+1}/{n_corridas}] — SEED: {seed}")
        print("█" * 60 + "\n")
        
        # Inyectar semilla en config.py
        update_config_seed(seed)
        
        # Ejecutar main.py en modo automático
        # Pasamos el argumento --clasificadores para que respete config.py
        cmd = [
            sys.executable, "main.py", 
            "--auto", 
            "--clasificadores", seleccion_str
        ]
        
        # Inyectar PYTHONUTF8 en las variables de entorno para que el proceso hijo
        # maneje correctamente los caracteres especiales (como ✓) en Windows.
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        t_corrida_ini = time.perf_counter()
        try:
            subprocess.run(cmd, cwd=base_dir, env=env, check=True)
        except subprocess.CalledProcessError:
            print(f"\n⚠ Error durante la ejecución de la corrida con seed {seed}.")
            print("  Se abortará el lote de simulaciones para revisar el error.")
            sys.exit(1)
        dt_corrida = time.perf_counter() - t_corrida_ini

        # main.py ya acumuló esta corrida en el JSON único
        # (resultados_benchmark.json) vía resultados_cache.guardar(seed=...).
        # No se renombra: todas las corridas del lote quedan en un solo archivo.
        json_unico = os.path.join(base_dir, "resultados", "resultados_benchmark.json")

        # ── Aviso por pantalla de fin de corrida ──────────────────────────
        restantes = n_corridas - (i + 1)
        dt_lote   = time.perf_counter() - t_inicio_lote
        print("\n" + "═" * 60)
        print(f"  ✓ CORRIDA {i+1}/{n_corridas} COMPLETADA  —  SEED {seed}")
        print(f"     Duración de la corrida : {dt_corrida/60:.1f} min")
        print(f"     Transcurrido del lote  : {dt_lote/60:.1f} min")
        print(f"     Corridas restantes     : {restantes}")
        if restantes > 0:
            eta = (dt_lote / (i + 1)) * restantes
            print(f"     ETA estimado restante  : {eta/60:.1f} min "
                  f"(~{eta/3600:.1f} h)")
        print("═" * 60)
        if os.path.exists(json_unico):
            print(f"  Acumulada en: {json_unico}")
        else:
            print(f"\n⚠ ADVERTENCIA: No se encontró el archivo {json_unico}")
            print("  Es posible que main.py no haya guardado los resultados.")
            
    t_total = time.perf_counter() - t_inicio_lote
    print("\n" + "=" * 60)
    print(f"  LOTE COMPLETADO: {n_corridas} corridas finalizadas.")
    print(f"  Tiempo total: {t_total/60:.1f} minutos.")
    print("=" * 60)

if __name__ == "__main__":
    main()
