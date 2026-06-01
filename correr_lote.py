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
import random
import subprocess
import re
import time

# Importamos config para saber qué clasificadores vamos a correr
import config
import main as main_script

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

def main():
    print("=" * 60)
    print("  ORQUESTADOR DE CORRIDAS MÚLTIPLES (BATCH RUNNER)")
    print("=" * 60)
    
    # Aviso sobre el uso de GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n  [INFO] Hardware: GPU detectada ({gpu_name}).")
            print("         ✓ ELM y Redes Neuronales utilizarán aceleración por hardware.")
        else:
            print("\n  [INFO] Hardware: No se detectó GPU compatible con CUDA.")
            print("         Se utilizará la CPU para todos los cálculos.")
    except ImportError:
        pass # Si torch no está instalado, simplemente ignoramos el aviso

    seleccion_str = seleccionar_clasificadores()
    
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
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"\n⚠ Error durante la ejecución de la corrida con seed {seed}.")
            print("  Se abortará el lote de simulaciones para revisar el error.")
            sys.exit(1)
            
        # Renombrar JSON resultante
        json_original = os.path.join(base_dir, "resultados", "resultados_benchmark.json")
        json_nuevo = os.path.join(base_dir, "resultados", f"resultados_seed_{seed}.json")
        
        if os.path.exists(json_original):
            if os.path.exists(json_nuevo):
                os.remove(json_nuevo)
            os.rename(json_original, json_nuevo)
            print(f"\n✓ Resultados guardados exitosamente en:")
            print(f"  {json_nuevo}")
        else:
            print(f"\n⚠ ADVERTENCIA: No se encontró el archivo {json_original}")
            print("  Es posible que main.py no haya guardado los resultados.")
            
    t_total = time.perf_counter() - t_inicio_lote
    print("\n" + "=" * 60)
    print(f"  LOTE COMPLETADO: {n_corridas} corridas finalizadas.")
    print(f"  Tiempo total: {t_total/60:.1f} minutos.")
    print("=" * 60)

if __name__ == "__main__":
    main()
