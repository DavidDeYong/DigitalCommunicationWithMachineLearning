"""
clasificadores/bayes.py
=======================
Detector Bayesiano para 16-QAM.

Es el clasificador de referencia (baseline). Con AWGN y símbolos equiprobables,
la regla de Máxima Verosimilitud equivale al vecino más cercano en el plano IQ.
No tiene parámetros entrenables → fit() es un no-op, pero se mantiene la
interfaz para que main.py lo trate igual que los demás.
"""

import numpy as np
from .base import ClasificadorBase
from canal.modulador import demodular_bayes_vectorizado, _TODOS_IQ, _TODOS_BITS


class ClasificadorBayes(ClasificadorBase):

    def __init__(self, A: float = 1.0):
        """
        Parameters
        ----------
        A : amplitud de la constelación (debe coincidir con la usada en la modulación)
        """
        super().__init__()
        self.A = A

    @property
    def nombre(self) -> str:
        return "Bayes (ML)"

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """No requiere entrenamiento. Marca como entrenado para mantener interfaz."""
        # ── FLOPs analíticos (estimado) ──────────────────────────────────
        # Distancia euclidiana a los 16 símbolos: por símbolo y dimensión
        # → resta + cuadrado + suma = 3 ops. Luego argmin (despreciable).
        d = X_train.shape[1]
        M = 16
        self.flops_inferencia    = float(M * 3 * d)
        self.n_parametros        = int(M * d)   # coordenadas de la constelación
        self.flops_tipo          = "estimado"
        # Bayes: solo almacena centroides de constelación, sin optimización iterativa
        self.flops_entrenamiento = 0.0

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        """
        Clasificación por vecino más cercano (equivalente a ML con AWGN).

        Parameters
        ----------
        X : shape (N, 2)  — símbolos IQ recibidos

        Returns
        -------
        etiquetas : shape (N,) int32   — índice del símbolo 0-15
        """
        constelacion = (self.A * _TODOS_IQ)[np.newaxis, :, :]  # (1, 16, 2)
        X_exp        = X[:, np.newaxis, :]                      # (N,  1, 2)
        distancias   = np.sum((X_exp - constelacion) ** 2, axis=2)  # (N, 16)
        return np.argmin(distancias, axis=1).astype(np.int32)

    def set_amplitud(self, A: float):
        """Actualiza la amplitud de la constelación (llamar antes de predict en cada Eb/N0)."""
        self.A = A
