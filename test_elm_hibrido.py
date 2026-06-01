"""
test_elm_hibrido.py
====================
Compara el modo híbrido GPU (float32 para H, float64 para el solve)
contra el fallback NumPy CPU (float64 puro).

Mide:
  - Tiempo de fit y predict
  - Diferencia numérica de beta (para verificar precisión)
"""

import numpy as np
import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clasificadores.elm_clf import _ELMCore

print("=" * 60)
print("  BENCHMARK HÍBRIDO ELM: GPU float32·H / CPU float64·solve")
print("=" * 60)
print(f"  PyTorch CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU                     : {torch.cuda.get_device_name(0)}")
print()

# ── Datos de prueba ────────────────────────────────────────────────────────────
N  = 100_000   # símbolos (como en la simulación real)
D  = 2         # I/Q
L  = 5_000     # neuronas ocultas ELM
C  = 10.0
SEED = 42

rng = np.random.default_rng(SEED)
X = rng.standard_normal((N, D))
y = rng.integers(0, 16, N, dtype=np.int32)

# ── MODO HÍBRIDO (GPU float32 / CPU float64) ──────────────────────────────────
print("[ 1/2 ] Modo HÍBRIDO GPU float32 + CPU float64 solve")
elm_gpu = _ELMCore(L, 'relu', C, seed=SEED, init_strategy='aleatorio')

t0 = time.perf_counter()
elm_gpu.fit(X.astype(np.float64), y)
t_fit_gpu = time.perf_counter() - t0

t0 = time.perf_counter()
preds_gpu = elm_gpu.predict(X.astype(np.float64))
t_pred_gpu = time.perf_counter() - t0

print(f"    _usar_gpu = {elm_gpu._usar_gpu}")
print(f"    fit       = {t_fit_gpu:.4f} s")
print(f"    predict   = {t_pred_gpu:.4f} s")
print()

# ── MODO FALLBACK NumPy CPU float64 ───────────────────────────────────────────
print("[ 2/2 ] Modo FALLBACK NumPy CPU float64 puro")

# Forzar fallback desactivando CUDA temporalmente
import torch
_orig = torch.cuda.is_available
torch.cuda.is_available = lambda: False

elm_cpu = _ELMCore(L, 'relu', C, seed=SEED, init_strategy='aleatorio')

t0 = time.perf_counter()
elm_cpu.fit(X.astype(np.float64), y)
t_fit_cpu = time.perf_counter() - t0

t0 = time.perf_counter()
preds_cpu = elm_cpu.predict(X.astype(np.float64))
t_pred_cpu = time.perf_counter() - t0

torch.cuda.is_available = _orig  # restaurar

print(f"    fit       = {t_fit_cpu:.4f} s")
print(f"    predict   = {t_pred_cpu:.4f} s")
print()

# ── COMPARACIÓN NUMÉRICA ──────────────────────────────────────────────────────
print("═" * 60)
print("  COMPARACIÓN DE RESULTADOS")
print("═" * 60)

# Diferencia en beta (el coeficiente aprendido)
diff_beta = np.abs(elm_gpu.beta - elm_cpu.beta)
print(f"  β diferencia máxima  : {diff_beta.max():.2e}")
print(f"  β diferencia media   : {diff_beta.mean():.2e}")
print(f"  β diferencia relativa: {(diff_beta / (np.abs(elm_cpu.beta) + 1e-15)).max():.2e}")
print()

# Coincidencia de predicciones
acuerdo = np.mean(preds_gpu == preds_cpu) * 100
print(f"  Predicciones idénticas: {acuerdo:.4f}%")
print()

# Resumen de velocidad
speedup_fit  = t_fit_cpu  / t_fit_gpu  if t_fit_gpu  > 0 else float('inf')
speedup_pred = t_pred_cpu / t_pred_gpu if t_pred_gpu > 0 else float('inf')
print("  RESUMEN DE VELOCIDAD:")
print(f"  {'Etapa':<15} {'GPU híbrido':>14} {'CPU float64':>14} {'Speedup':>10}")
print("  " + "-" * 56)
print(f"  {'fit':<15} {t_fit_gpu:>12.4f} s {t_fit_cpu:>12.4f} s {speedup_fit:>9.1f}x")
print(f"  {'predict':<15} {t_pred_gpu:>12.4f} s {t_pred_cpu:>12.4f} s {speedup_pred:>9.1f}x")
print()
print("  ✓ Test completado.")
