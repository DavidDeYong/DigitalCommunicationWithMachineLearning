# Guía de Ejecución: Benchmark ML para 16-QAM

Este documento describe la estructura básica del proyecto, cómo configurarlo y cómo lanzar las simulaciones, ya sea de forma manual (corrida única) o automatizada (múltiples corridas con distintas semillas).

---

## 1. ¿Dónde se configuran los parámetros?

**El archivo `config.py` es el "cerebro" de toda la configuración.**  
Absolutamente todos los parámetros del experimento se cambian desde ahí. No hace falta modificar el código fuente de los clasificadores para cambiar el experimento.

Parámetros típicos que vas a querer modificar en `config.py`:
- `EB_N0_RANGE_DB`: Puntos del barrido de ruido.
- `MODO_ENTRENAMIENTO` y `MODO_TEST`: Para definir si la fuente y el entrenamiento se hacen con datos reales (voz) o piloto sintético (recomendado).
- Ajustes de hiperparámetros (ej. `ELM_HIDDEN_DEFAULT`, `SVM_C_DEFAULT`).

---

## 2. ¿Qué archivos ejecutar desde tu IDE?

Tienes dos formas principales de utilizar este código dependiendo de tu objetivo. En tu IDE (PyCharm, VS Code, Spyder, etc.), debes seleccionar y ejecutar uno de los siguientes archivos:

### Opción A: Corrida interactiva (Ideal para pruebas y debugging)
**Archivo a ejecutar:** `main.py`

- Al correr este script, la consola te irá haciendo preguntas (ej. qué clasificadores querés correr, si deseas usar el caché de hiperparámetros).
- Es ideal cuando estás programando, probando un nuevo clasificador o viendo gráficas de forma interactiva.
- Guarda un único archivo de resultados llamado `resultados_benchmark.json` en la carpeta `resultados/`.

### Opción B: Automatización y generación de datos (Ideal para dejar la PC corriendo)
**Archivo a ejecutar:** `correr_lote.py`

- Este es el orquestador automático. Al ejecutarlo, te preguntará **únicamente**: "¿Cuántas corridas independientes deseas ejecutar?".
- Él mismo generará las semillas aleatorias, modificará dinámicamente `config.py` para cada corrida y llamará a `main.py` de forma invisible.
- Por cada corrida, generará un archivo independiente en la carpeta `resultados/` con el nombre `resultados_seed_XXXXX.json`.
- **Ideal para:** Dejar la computadora trabajando a la noche (usando la GPU) para generar estadísticas sobre 50 o 100 semillas diferentes.

---

## 3. Configuración del IDE y la GPU

1. **Intérprete de Python:** Asegúrate de que tu IDE esté apuntando al entorno de Python donde instalamos PyTorch con soporte para CUDA (típicamente Python 3.13 de Windows Store).
2. **Detección Automática:** El código (`elm_clf.py` y redes neuronales) detecta automáticamente si tu placa NVIDIA (RTX 4060) está disponible y la utiliza para acelerar el entrenamiento en x70 veces.
3. **Fallback a CPU:** Si el día de mañana te llevas este código a una notebook sin GPU dedicada, el código lo detectará y correrá sobre la CPU sin mostrar ningún error (aunque tardará más). No necesitas cambiar nada en el código.

---

## 4. ¿Dónde encuentro los resultados?

Todas las salidas se guardan en carpetas dentro del directorio del proyecto:

- `resultados/`: Aquí van los JSON con las métricas puras y las gráficas en formato PNG.
- `hiperparametros/`: El caché del GridSearch. Si en algún momento querés que los clasificadores vuelvan a buscar sus parámetros óptimos, simplemente borrá los `.json` de esta carpeta.
- `modelos/`: Si tenés activada la opción `GUARDAR_MODELOS` en `config.py`, aquí se guardarán los pesos entrenados de los algoritmos.
