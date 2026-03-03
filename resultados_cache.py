"""
resultados_cache.py
===================
Sistema de persistencia y fusión de resultados del benchmark.

Permite correr solo un subconjunto de clasificadores y fusionar los
resultados con los de corridas anteriores. El archivo final siempre
contiene un resultado por clasificador — si un clasificador fue
re-testeado, el valor nuevo reemplaza al anterior.

Archivo: <DIR_RESULTADOS>/resultados_benchmark.json
"""

import os, json
import numpy as np
from datetime import datetime

NOMBRE_ARCHIVO = "resultados_benchmark.json"


# ---------------------------------------------------------------------------
# Utilidades de serialización
# ---------------------------------------------------------------------------

def _a_python(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)):  return obj.tolist()
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    return obj

def _serializar(reg):
    out = {}
    for k, v in reg.items():
        out[k] = [_a_python(x) for x in v] if isinstance(v, list) else _a_python(v)
    return out


# ---------------------------------------------------------------------------
# Guardar / cargar
# ---------------------------------------------------------------------------

def guardar(registros: list, dir_cache: str, config_snap: dict = None) -> str:
    """
    Fusiona los registros nuevos con los existentes y guarda en disco.
    Los clasificadores re-testeados reemplazan sus resultados anteriores.
    Los clasificadores no testeados en esta corrida se conservan intactos.
    """
    os.makedirs(dir_cache, exist_ok=True)
    ruta = os.path.join(dir_cache, NOMBRE_ARCHIVO)

    # Cargar existentes
    datos = _cargar_raw(ruta)

    # Fusionar: nuevos reemplazan a anteriores del mismo nombre
    for reg in registros:
        datos["registros"][reg["nombre"]] = _serializar(reg)
        datos["fechas"][reg["nombre"]]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    datos["fecha_ultima_actualizacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if config_snap:
        datos["config_snapshot"] = config_snap

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    print(f"\n  [Resultados] Guardados en '{ruta}'")
    print(f"  [Resultados] Clasificadores almacenados: "
          f"{list(datos['registros'].keys())}")
    return ruta


def cargar(dir_cache: str) -> dict:
    """
    Carga los resultados guardados.
    Devuelve dict {nombre_clf: registro} o {} si no existe archivo.
    """
    ruta = os.path.join(dir_cache, NOMBRE_ARCHIVO)
    datos = _cargar_raw(ruta)
    return datos["registros"]


def existe(dir_cache: str) -> bool:
    return os.path.exists(os.path.join(dir_cache, NOMBRE_ARCHIVO))


def _cargar_raw(ruta: str) -> dict:
    if os.path.exists(ruta):
        with open(ruta, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        # Asegurar estructura completa en archivos antiguos
        datos.setdefault("registros", {})
        datos.setdefault("fechas",    {})
        return datos
    return {"registros": {}, "fechas": {}, "fecha_ultima_actualizacion": None,
            "config_snapshot": {}}


# ---------------------------------------------------------------------------
# Resumen en consola
# ---------------------------------------------------------------------------

def imprimir_resumen(dir_cache: str) -> None:
    if not existe(dir_cache):
        print("  [Resultados] No hay resultados guardados todavía.")
        return

    ruta  = os.path.join(dir_cache, NOMBRE_ARCHIVO)
    datos = _cargar_raw(ruta)

    print(f"\n  Resultados guardados en '{ruta}':")
    print(f"  Última actualización: {datos.get('fecha_ultima_actualizacion', '?')}\n")
    print(f"  {'Clasificador':<22} {'Fecha de corrida':<22} {'Puntos Eb/N0':>13}")
    print("  " + "-"*60)

    fechas = datos.get("fechas", {})
    for nombre, reg in datos["registros"].items():
        n_puntos = len(reg.get("Eb_N0_dB", []))
        fecha    = fechas.get(nombre, "desconocida")
        print(f"  {nombre:<22} {fecha:<22} {n_puntos:>13}")


# ---------------------------------------------------------------------------
# Reconstruir lista de registros completa (guardados + nuevos)
# ---------------------------------------------------------------------------

def registros_completos(nuevos: list, dir_cache: str) -> list:
    """
    Devuelve una lista de registros que combina:
      - Los registros de esta corrida (nuevos)
      - Los registros de corridas anteriores para los clasificadores no corridos

    El orden respeta el orden original de la lista de clasificadores.
    Los nuevos tienen precedencia sobre los guardados.
    """
    guardados = cargar(dir_cache)
    nuevos_nombres = {r["nombre"] for r in nuevos}

    # Añadir guardados que no fueron re-testeados en esta corrida
    adicionales = []
    for nombre, reg in guardados.items():
        if nombre not in nuevos_nombres:
            adicionales.append(reg)

    # Orden: todos los nuevos primero, luego los históricos
    # Pero reordenar según el orden canónico si está disponible
    todos = nuevos + adicionales
    return todos
