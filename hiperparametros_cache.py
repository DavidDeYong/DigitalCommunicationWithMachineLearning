"""
hiperparametros_cache.py
========================
Sistema de caché para hiperparámetros óptimos encontrados por GridSearch.

Funcionamiento:
  - Tras cada GridSearch exitoso, guarda los parámetros en un archivo JSON.
  - En la próxima corrida, si el JSON existe y USAR_CACHE=True en config.py,
    carga los parámetros y saltea el GridSearch completamente.
  - El nombre del archivo identifica al clasificador, así cada uno tiene su propio cache.

Ubicación del archivo: DIR_HIPERPARAMETROS definido en config.py
  (por defecto: "hiperparametros/")

Formato del JSON:
  {
    "clasificador": "SVM RBF",
    "fecha"       : "2026-02-20 15:32:10",
    "params"      : {"C": 10, "gamma": 0.1},
    "cv_accuracy" : 0.9821
  }
"""

import os
import json
from datetime import datetime


def _ruta_cache(nombre_clasificador: str, dir_cache: str) -> str:
    """Devuelve la ruta al archivo JSON para un clasificador dado."""
    nombre_archivo = nombre_clasificador.replace(" ", "_").lower() + "_hiperparams.json"
    return os.path.join(dir_cache, nombre_archivo)


def guardar(
    nombre_clasificador: str,
    params:              dict,
    dir_cache:           str,
    cv_accuracy:         float = None,
) -> None:
    """
    Guarda los hiperparámetros óptimos en disco.

    Parameters
    ----------
    nombre_clasificador : str   nombre del clasificador (e.g. "SVM RBF")
    params              : dict  hiperparámetros óptimos (e.g. {'C': 10, 'gamma': 0.1})
    dir_cache           : str   directorio donde guardar el JSON
    cv_accuracy         : float accuracy de validación cruzada (opcional, informativo)
    """
    os.makedirs(dir_cache, exist_ok=True)
    ruta = _ruta_cache(nombre_clasificador, dir_cache)

    # Convertir valores numpy a tipos Python nativos para serialización JSON
    params_serializables = _convertir_tipos(params)

    contenido = {
        "clasificador": nombre_clasificador,
        "fecha"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params"      : params_serializables,
        "cv_accuracy" : float(cv_accuracy) if cv_accuracy is not None else None,
    }

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False)

    print(f"  [Cache] Hiperparámetros guardados en '{ruta}'")


def cargar(
    nombre_clasificador: str,
    dir_cache:           str,
) -> dict | None:
    """
    Carga los hiperparámetros desde disco si el archivo existe.

    Returns
    -------
    dict con los parámetros óptimos, o None si no existe el cache.
    """
    ruta = _ruta_cache(nombre_clasificador, dir_cache)

    if not os.path.exists(ruta):
        return None

    with open(ruta, 'r', encoding='utf-8') as f:
        contenido = json.load(f)

    params = contenido.get("params", {})
    fecha  = contenido.get("fecha", "desconocida")
    acc    = contenido.get("cv_accuracy")

    acc_str = f"  |  CV accuracy = {acc:.4f}" if acc is not None else ""
    print(f"  [Cache] ✓ Hiperparámetros cargados desde cache (guardado: {fecha}{acc_str})")
    print(f"  [Cache]   Params: {params}")

    return params


def existe(nombre_clasificador: str, dir_cache: str) -> bool:
    """Retorna True si existe cache para el clasificador dado."""
    return os.path.exists(_ruta_cache(nombre_clasificador, dir_cache))


def imprimir_resumen_cache(dir_cache: str) -> None:
    """
    Imprime un resumen de todos los caches disponibles.
    Útil para ver de un vistazo qué hiperparámetros óptimos ya fueron encontrados.
    """
    if not os.path.exists(dir_cache):
        print(f"  [Cache] No existe el directorio '{dir_cache}'. Aún no se han guardado hiperparámetros.")
        return

    archivos = [f for f in os.listdir(dir_cache) if f.endswith('_hiperparams.json')]

    if not archivos:
        print(f"  [Cache] No hay hiperparámetros guardados en '{dir_cache}'.")
        return

    print(f"\n  ── Hiperparámetros óptimos en cache ({dir_cache}) ──")
    print(f"  {'Clasificador':<20} {'Parámetros':<35} {'CV Accuracy':>12}  {'Fecha'}")
    print("  " + "-"*90)

    for archivo in sorted(archivos):
        ruta = os.path.join(dir_cache, archivo)
        with open(ruta, 'r', encoding='utf-8') as f:
            contenido = json.load(f)

        nombre = contenido.get("clasificador", archivo)
        params = contenido.get("params", {})
        fecha  = contenido.get("fecha", "?")
        acc    = contenido.get("cv_accuracy")
        acc_str = f"{acc:.4f}" if acc is not None else "  N/A"

        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {nombre:<20} {params_str:<35} {acc_str:>12}  {fecha}")


def _convertir_tipos(obj):
    """Convierte tipos numpy/especiales a tipos Python nativos para JSON."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convertir_tipos(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
