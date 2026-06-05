"""
config.py
=========
Punto único de configuración del benchmark.
Todos los parámetros del sistema se definen aquí.
main.py y el resto de los módulos importan desde este archivo.
"""

# ---------------------------------------------------------------------------
# FUENTE DE INFORMACIÓN
# ---------------------------------------------------------------------------
RUTA_AUDIO      = "voz.wav"     # Ruta al archivo WAV. Si no existe, se usa el fallback sintético.
MU              = 255           # Parámetro de la Ley µ (estándar telecomunicaciones)
BITS_CUANT      = 8             # Bits de cuantificación → 256 niveles → 4 bits/símbolo 16-QAM
FS_RESAMPLE     = 44100         # Frecuencia de muestreo objetivo (Hz)

# ---------------------------------------------------------------------------
# MODULACIÓN
# ---------------------------------------------------------------------------
# 16-QAM: 4 bits por símbolo, constelación en plano I/Q con mapeo Gray
BITS_POR_SIMBOLO = 4
M_QAM            = 16

# ---------------------------------------------------------------------------
# BARRIDO DE Eb/N0
# ---------------------------------------------------------------------------
EB_N0_RANGE_DB = [0, 2, 4, 6, 8, 10, 12]   # dB

# ---------------------------------------------------------------------------
# TAMAÑOS DE CONJUNTOS (en símbolos 16-QAM)
# ---------------------------------------------------------------------------
N_TRAIN    = 100_000   # Símbolos para la fase de entrenamiento (fijo en todo el barrido)
N_TEST     = 100_000   # Símbolos de test base — se amplía adaptativamente si es necesario
N_MAX_TEST = 2_000_000 # Límite práctico para N_TEST adaptativo (tiempo de cómputo)
# Nota: N_TEST se incrementa automáticamente en puntos de alto Eb/N0 para
# garantizar al menos 100 errores contados. Si el N necesario supera N_MAX_TEST,
# el punto se grafica con marcador de "estadística insuficiente" y se reporta
# cuántos símbolos se necesitarían para hacerlo confiable.

# ---------------------------------------------------------------------------
# REPRODUCIBILIDAD
# ---------------------------------------------------------------------------
SEED = 96598

# ---------------------------------------------------------------------------
# VALIDACIÓN DEL SIMULADOR
# ---------------------------------------------------------------------------
# El sistema grafica BER_Bayes_simulada vs BER_teórica y pausa para que
# el usuario confirme visualmente antes de continuar con ML.
VALIDAR_SIMULADOR = False    # False para saltar la validación (no recomendado)

# ---------------------------------------------------------------------------
# SELECCIÓN Y CONFIGURACIÓN DE CLASIFICADORES
# ---------------------------------------------------------------------------
# HARDWARE SELECTION
# ---------------------------------------------------------------------------
# FORCE_CPU: if True, all classifiers (including neural nets) run on CPU even if a GPU is available.
FORCE_CPU = False
# New flags for GPU SVM and operation count metrics
USE_GPU_SVM = True  # When True, attempt to run SVM on GPU via ThunderSVM if available
COMPUTE_OPERATION_COUNTS = True  # Enable FLOP/parameter counting for PyTorch models


# DISPLAY SETTINGS
# SHOW_PLOTS: if True, figures are displayed (blocking). Set to False for batch runs.
SHOW_PLOTS = False
# Define qué clasificadores se ejecutan en modo automático (como al usar correr_lote.py)
# "A"       -> Ejecuta TODOS los clasificadores disponibles.
# "1,2,5"   -> Ejecuta solo los seleccionados por su índice (ej: Bayes, SVM, ELM).
CLASIFICADORES_A_CORRER = "A"

OPTIMIZAR_HIPERPARAMETROS = False   # True: GridSearch automático | False: usa valores fijos

# SVM
SVM_C_DEFAULT     = 10
SVM_GAMMA_DEFAULT = 0.1
SVM_C_GRID        = [0.1, 1, 10, 100]
SVM_GAMMA_GRID    = [0.01, 0.1, 1, 'scale']
SVM_CV_FOLDS      = 5

# KNN
KNN_K_DEFAULT = 15
KNN_K_GRID    = [3, 7, 15, 25, 51]

# SGD-Nystroem
SGD_N_COMPONENTS_DEFAULT = 500
SGD_GAMMA_DEFAULT        = 1.0
SGD_N_COMPONENTS_GRID    = [100, 300, 500, 1000]
SGD_GAMMA_GRID           = [0.1, 0.5, 1.0, 2.0]

# ELM (Extreme Learning Machine)
ELM_HIDDEN_DEFAULT     = 1000
ELM_C_DEFAULT          = 10.0
ELM_HIDDEN_GRID        = [500, 1000, 2000, 5000, 10000]
ELM_C_GRID             = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ELM_ACTIVATION_GRID    = ['relu', 'tanh']   # se evalúan en GridSearch

# Logistic Regression
LR_C_DEFAULT  = 1.0
LR_C_GRID     = [0.01, 0.1, 1.0, 10.0, 100.0]
LR_CV_FOLDS   = 3
LR_MAX_ITER   = 1000

# Random Forest
RF_N_TREES_DEFAULT  = 100
RF_MAX_DEPTH_DEFAULT = 8
RF_N_TREES_GRID     = [50, 100, 200]
RF_MAX_DEPTH_GRID   = [6, 8, 12, None]

# Red Neuronal MLP (simple)
NN_MLP_HIDDEN       = (128, 64)     # capas ocultas
NN_MLP_EPOCHS       = 100
NN_MLP_BATCH_SIZE   = 512
NN_MLP_LR           = 1e-3          # learning rate inicial del optimizador Adam
NN_MLP_LR_PATIENCE  = 10            # épocas sin mejora antes de reducir LR
NN_MLP_LR_FACTOR    = 0.5           # factor de reducción del LR (nuevo_LR = LR * factor)
NN_MLP_LR_MIN       = 1e-5          # LR mínimo — el scheduler no reduce por debajo de esto

# Red Neuronal Profunda
NN_DEEP_HIDDEN      = (256, 128, 64, 32)
NN_DEEP_EPOCHS      = 200
NN_DEEP_BATCH_SIZE  = 512
NN_DEEP_LR          = 1e-3          # learning rate inicial
NN_DEEP_LR_PATIENCE = 15            # más paciencia porque tiene más épocas
NN_DEEP_LR_FACTOR   = 0.5
NN_DEEP_LR_MIN      = 1e-5

# ---------------------------------------------------------------------------
# PROTOCOLO DE ENTRENAMIENTO Y TEST
# ---------------------------------------------------------------------------
# MODO_ENTRENAMIENTO controla el origen de los bits usados para entrenar:
#
#   'piloto'  → Secuencia pseudoaleatoria sintética (equiprobable por construcción).
#               Emula el protocolo real: el transmisor envía una secuencia piloto
#               conocida antes de los datos. El receptor la recibe con el canal
#               embebido y ajusta el clasificador. Garantiza equiprobabilidad
#               independientemente de la fuente de información.
#
#   'fuente'  → Segmento del audio de voz (comportamiento anterior).
#               Útil para análisis de sensibilidad a la distribución de la fuente.
#
# MODO_TEST controla el origen de los bits usados para evaluar BER:
#
#   'fuente'  → Bitstream de voz real (condición realista post-entrenamiento).
#               El clasificador ya entrenado demodula el contenido real.
#
#   'piloto'  → Secuencia sintética equiprobable (igual que el entrenamiento).
#               Útil para aislar el efecto del canal de la distribución de la fuente.
#
# Configuración recomendada para el artículo:
#   MODO_ENTRENAMIENTO = 'piloto'   → protocolo realista, equiprobabilidad garantizada
#   MODO_TEST          = 'fuente'   → evaluación sobre voz real

MODO_ENTRENAMIENTO = 'piloto'   # 'piloto' | 'fuente'
MODO_TEST          = 'fuente'   # 'fuente' | 'piloto'
PILOTO_SEED        = 3010       # Semilla para la secuencia piloto (fija TX y RX)

# ---------------------------------------------------------------------------
# SCRAMBLING DE BITS
# ---------------------------------------------------------------------------
# El scrambling XOR con una secuencia pseudoaleatoria garantiza P(1) ≈ 0.5
# independientemente de la distribución de la fuente (voz + Ley µ).
# Es la práctica estándar en sistemas de telecomunicaciones reales.
#
# USAR_SCRAMBLING = True  → bits equiprobables, condición ideal para Bayes
# USAR_SCRAMBLING = False → bits de la fuente sin modificar (condición real)
#
# Para el artículo se recomienda presentar ambos casos y discutir la diferencia.
USAR_SCRAMBLING  = True
SCRAMBLING_SEED  = 42      # Semilla del LFSR (debe ser igual en TX y RX)
SCRAMBLING_POLY  = 0x400007  # Polinomio LFSR de 23 bits (período 2²³-1 ≈ 8M)

# ---------------------------------------------------------------------------
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ---------------------------------------------------------------------------
# Punto del barrido usado para el GridSearch.
# Se elige un Eb/N0 intermedio-bajo donde BER ~ 10^-2 a 10^-3,
# que para 16-QAM en AWGN corresponde a ~8 dB.
# Optimizar en este punto garantiza parámetros robustos en toda la región útil.
EB_N0_PARA_OPTIMIZACION_DB = 8   # debe estar incluido en EB_N0_RANGE_DB

# ---------------------------------------------------------------------------
# CACHÉ DE HIPERPARÁMETROS
# ---------------------------------------------------------------------------
# Si USAR_CACHE_HIPERPARAMETROS = True y ya existe un JSON con los parámetros
# óptimos de una corrida anterior, el GridSearch se saltea completamente.
# Borrar los archivos JSON en DIR_HIPERPARAMETROS para forzar una nueva búsqueda.
USAR_CACHE_HIPERPARAMETROS = True

# ---------------------------------------------------------------------------
# SALIDAS
# ---------------------------------------------------------------------------
# Los directorios se construyen como rutas absolutas relativas a la ubicación
# de este archivo, para que funcionen correctamente en Windows / Google Drive.
import os as _os
_BASE = _os.path.dirname(_os.path.abspath(__file__))

GUARDAR_MODELOS   = True
DIR_MODELOS       = _os.path.join(_BASE, "modelos")
DIR_RESULTADOS    = _os.path.join(_BASE, "resultados")
DIR_HIPERPARAMETROS = _os.path.join(_BASE, "hiperparametros")
GUARDAR_GRAFICAS  = True
