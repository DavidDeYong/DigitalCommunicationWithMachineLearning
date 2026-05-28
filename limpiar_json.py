"""
limpiar_json.py
===============
Limpia el resultados_benchmark.json eliminando clasificadores
que fueron copiados por error de corridas anteriores.

Ejecutar UNA SOLA VEZ en la carpeta del proyecto.
"""
import json, os

RUTA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "resultados", "resultados_benchmark.json")

# Clasificadores que NO se corrieron en la corrida seed=42
# (completar con los que realmente corriste en esa sesión)
SEED_A_LIMPIAR   = 42
CLFS_NO_CORRIDOS = ["KNN (k=15)", "Random Forest"]

with open(RUTA_JSON, 'r', encoding='utf-8') as f:
    datos = json.load(f)

for corrida in datos["corridas"]:
    if corrida.get("seed") == SEED_A_LIMPIAR:
        for clf in CLFS_NO_CORRIDOS:
            if clf in corrida["registros"]:
                del corrida["registros"][clf]
                print(f"  Eliminado '{clf}' de corrida seed={SEED_A_LIMPIAR}")
        print(f"  Clasificadores restantes: {list(corrida['registros'].keys())}")
        break

with open(RUTA_JSON, 'w', encoding='utf-8') as f:
    json.dump(datos, f, indent=2, ensure_ascii=False)

print("\n  JSON limpio guardado.")
