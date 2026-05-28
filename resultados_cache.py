"""
resultados_cache.py
===================
Sistema de persistencia de resultados del benchmark con soporte para
múltiples corridas independientes (distintos seeds).

Esquema del JSON
----------------
{
  "fecha_ultima_actualizacion": "2026-03-30 16:16:01",
  "n_corridas": 3,
  "corridas": [
    {
      "seed"           : 42,
      "corrida_id"     : 1,
      "fecha"          : "2026-03-30 16:16:01",
      "config_snapshot": {...},
      "registros"      : {
          "Bayes (ML)": { registro completo },
          "SVM RBF"   : { registro completo },
          ...
      }
    },
    {
      "seed": 123,
      "corrida_id": 2,
      ...
    }
  ]
}

Comportamiento
--------------
- Cada seed genera una entrada independiente en "corridas".
- Si el mismo seed se corre dos veces, la segunda reemplaza a la primera
  (permite re-correr con correcciones sin duplicar datos).
- Si en una corrida solo se corren algunos clasificadores, los restantes
  se completan desde la corrida anterior del mismo seed (si existe) o
  desde la corrida más reciente de cualquier seed.
- Las funciones que usa main.py mantienen compatibilidad hacia atrás;
  solo se agrega el parámetro 'seed' donde corresponde.

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
# Carga interna del JSON completo
# ---------------------------------------------------------------------------

def _cargar_raw(ruta: str) -> dict:
    """
    Carga el JSON y normaliza tanto el esquema nuevo (corridas[])
    como el esquema viejo (registros{}) para compatibilidad hacia atrás.
    """
    if not os.path.exists(ruta):
        return {"corridas": [], "fecha_ultima_actualizacion": None}

    with open(ruta, 'r', encoding='utf-8') as f:
        datos = json.load(f)

    # ── Migración automática del esquema antiguo ──────────────────────────
    # El esquema viejo tenía "registros" y "fechas" en el nivel raíz.
    if "registros" in datos and "corridas" not in datos:
        print("  [Cache] Migrando JSON al esquema multi-corrida...")
        registros_viejos = datos.get("registros", {})
        fecha_vieja      = datos.get("fecha_ultima_actualizacion",
                                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        datos = {
            "corridas": [
                {
                    "seed"           : 0,   # seed desconocido → usar 0
                    "corrida_id"     : 1,
                    "fecha"          : fecha_vieja,
                    "config_snapshot": datos.get("config_snapshot", {}),
                    "registros"      : registros_viejos,
                }
            ],
            "fecha_ultima_actualizacion": fecha_vieja,
        }

    datos.setdefault("corridas", [])
    datos.setdefault("fecha_ultima_actualizacion", None)
    return datos


# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------

def guardar(registros: list, dir_cache: str,
            config_snap: dict = None, seed: int = None) -> str:
    """
    Guarda los registros de esta corrida acumulándolos en el JSON.

    - Si ya existe una corrida con el mismo seed, la reemplaza.
    - Si es un seed nuevo, agrega una entrada nueva.
    - Los clasificadores no corridos en esta sesión se completan desde
      la corrida anterior del mismo seed (o la más reciente).

    Parameters
    ----------
    registros   : lista de registros de esta corrida
    dir_cache   : directorio de resultados
    config_snap : snapshot de parámetros de config (opcional)
    seed        : semilla usada en esta corrida (opcional, pero recomendado)
    """
    os.makedirs(dir_cache, exist_ok=True)
    ruta  = os.path.join(dir_cache, NOMBRE_ARCHIVO)
    datos = _cargar_raw(ruta)

    ahora      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    seed_actual = seed if seed is not None else -1

    # ── Buscar si ya existe una corrida con este seed ─────────────────────
    idx_existente = None
    for i, corrida in enumerate(datos["corridas"]):
        if corrida.get("seed") == seed_actual:
            idx_existente = i
            break

    # ── Construir registros serializados — SOLO los de esta corrida ────────
    # Cada corrida guarda únicamente los clasificadores que realmente se
    # ejecutaron. No se mezclan resultados de otras corridas ni seeds.
    # El completado con corridas anteriores ocurre solo al graficar.
    registros_esta_corrida = {reg["nombre"]: _serializar(reg) for reg in registros}

    # ── Construir entrada de corrida ──────────────────────────────────────
    corrida_id = (idx_existente + 1) if idx_existente is not None \
                 else len(datos["corridas"]) + 1

    nueva_corrida = {
        "seed"           : seed_actual,
        "corrida_id"     : corrida_id,
        "fecha"          : ahora,
        "config_snapshot": config_snap or {},
        "registros"      : registros_esta_corrida,
    }


    # ── Insertar o reemplazar ─────────────────────────────────────────────
    if idx_existente is not None:
        datos["corridas"][idx_existente] = nueva_corrida
        accion = f"actualizada (seed={seed_actual})"
    else:
        datos["corridas"].append(nueva_corrida)
        accion = f"nueva (seed={seed_actual})"

    datos["fecha_ultima_actualizacion"] = ahora

    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    n_corridas = len(datos["corridas"])
    print(f"\n  [Resultados] Guardados en '{ruta}'")
    print(f"  [Resultados] Corrida {accion} — Total corridas acumuladas: {n_corridas}")
    print(f"  [Resultados] Clasificadores en esta corrida: "
          f"{list(registros_esta_corrida.keys())}")
    return ruta


# ---------------------------------------------------------------------------
# Cargar — devuelve los registros de la corrida más reciente
# ---------------------------------------------------------------------------

def cargar(dir_cache: str, seed: int = None) -> dict:
    """
    Carga los registros de una corrida específica (por seed) o de la
    más reciente si no se especifica seed.

    Returns
    -------
    dict {nombre_clf: registro}
    """
    ruta  = os.path.join(dir_cache, NOMBRE_ARCHIVO)
    datos = _cargar_raw(ruta)

    if not datos["corridas"]:
        return {}

    if seed is not None:
        for corrida in datos["corridas"]:
            if corrida.get("seed") == seed:
                return corrida.get("registros", {})
        return {}  # seed no encontrado

    # Sin seed especificado → corrida más reciente
    return datos["corridas"][-1].get("registros", {})


def cargar_todas_corridas(dir_cache: str) -> list:
    """
    Devuelve la lista completa de corridas acumuladas.
    Cada elemento es un dict con: seed, corrida_id, fecha, registros.

    Útil para el análisis estadístico multi-corrida.
    """
    ruta  = os.path.join(dir_cache, NOMBRE_ARCHIVO)
    datos = _cargar_raw(ruta)
    return datos.get("corridas", [])


def existe(dir_cache: str) -> bool:
    return os.path.exists(os.path.join(dir_cache, NOMBRE_ARCHIVO))


# ---------------------------------------------------------------------------
# Resumen en consola
# ---------------------------------------------------------------------------

def imprimir_resumen(dir_cache: str) -> None:
    if not existe(dir_cache):
        print("  [Resultados] No hay resultados guardados todavía.")
        return

    ruta  = os.path.join(dir_cache, NOMBRE_ARCHIVO)
    datos = _cargar_raw(ruta)
    corridas = datos.get("corridas", [])

    if not corridas:
        print("  [Resultados] No hay corridas guardadas todavía.")
        return

    print(f"\n  Resultados guardados en '{ruta}':")
    print(f"  Última actualización: {datos.get('fecha_ultima_actualizacion', '?')}")
    print(f"  Corridas acumuladas : {len(corridas)}\n")

    # Mostrar resumen de la corrida más reciente
    ultima = corridas[-1]
    print(f"  Corrida más reciente — seed={ultima.get('seed', '?')}  "
          f"fecha={ultima.get('fecha', '?')}")
    print(f"\n  {'Clasificador':<22} {'Fecha de corrida':<22} {'Puntos Eb/N0':>13}")
    print("  " + "-"*60)

    for nombre, reg in ultima.get("registros", {}).items():
        n_puntos = len(reg.get("Eb_N0_dB", []))
        print(f"  {nombre:<22} {ultima.get('fecha', '?'):<22} {n_puntos:>13}")

    # Resumen de todas las corridas
    if len(corridas) > 1:
        print(f"\n  Historial de corridas:")
        print(f"  {'ID':>4}  {'Seed':>6}  {'Fecha':<22}  {'Clasificadores'}")
        print("  " + "-"*70)
        for c in corridas:
            clfs = list(c.get("registros", {}).keys())
            clfs_str = ", ".join(clfs[:4])
            if len(clfs) > 4:
                clfs_str += f" ... (+{len(clfs)-4})"
            print(f"  {c.get('corrida_id', '?'):>4}  "
                  f"{c.get('seed', '?'):>6}  "
                  f"{c.get('fecha', '?'):<22}  {clfs_str}")


# ---------------------------------------------------------------------------
# Reconstruir lista de registros para las gráficas de esta sesión
# ---------------------------------------------------------------------------

def registros_completos(nuevos: list, dir_cache: str, seed: int = None) -> list:
    """
    Combina los registros nuevos con los guardados de la corrida más
    reciente, para que las gráficas incluyan todos los clasificadores.

    Los nuevos tienen precedencia sobre los guardados.
    """
    guardados      = cargar(dir_cache, seed=seed)
    nuevos_nombres = {r["nombre"] for r in nuevos}

    adicionales = []
    for nombre, reg in guardados.items():
        if nombre not in nuevos_nombres:
            adicionales.append(reg)

    return nuevos + adicionales
