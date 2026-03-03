"""
clasificadores/base.py
======================
Clase base abstracta que define la interfaz común para todos los clasificadores.

Todo clasificador del benchmark debe heredar de ClasificadorBase e implementar:
  - fit()
  - predict()
  - optimizar_hiperparametros()  (puede ser no-op si no aplica)
  - nombre (property)

Esta interfaz uniforme permite que main.py trate todos los clasificadores
de forma intercambiable, sin conocer sus detalles internos.
"""

from abc import ABC, abstractmethod
import numpy as np
import time


class ClasificadorBase(ABC):
    """
    Interfaz común para todos los clasificadores del benchmark.

    Atributos internos gestionados automáticamente:
      _tiempo_entrenamiento : float   segundos de la última llamada a fit()
      _tiempo_inferencia_1  : float   segundos para clasificar 1 símbolo
      _entrenado            : bool
    """

    def __init__(self):
        self._tiempo_entrenamiento: float = 0.0
        self._tiempo_inferencia_1:  float = 0.0
        self._entrenado:            bool  = False

    # ------------------------------------------------------------------
    # Métodos abstractos — deben implementarse en cada subclase
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def nombre(self) -> str:
        """Nombre legible del clasificador para gráficas y tablas."""
        ...

    @abstractmethod
    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Lógica interna de entrenamiento. No medir tiempo aquí.

        Parameters
        ----------
        X_train : shape (N, 2)   — símbolos IQ recibidos
        y_train : shape (N,)     — etiquetas enteras 0-15
        """
        ...

    @abstractmethod
    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        """
        Lógica interna de predicción. No medir tiempo aquí.

        Parameters
        ----------
        X : shape (N, 2)

        Returns
        -------
        etiquetas : shape (N,) int
        """
        ...

    def optimizar_hiperparametros(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> dict:
        """
        Búsqueda de hiperparámetros óptimos. Implementación por defecto: no-op.
        Subclases con GridSearch sobreescriben este método.

        Returns
        -------
        mejores_params : dict
        """
        return {}

    # ------------------------------------------------------------------
    # Métodos públicos con instrumentación de tiempo (no sobreescribir)
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClasificadorBase":
        """
        Entrena el clasificador y mide el tiempo de entrenamiento.

        Returns self para encadenamiento.
        """
        t0 = time.perf_counter()
        self._fit_interno(X_train, y_train)
        self._tiempo_entrenamiento = time.perf_counter() - t0
        self._entrenado = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice etiquetas para X. Mide el tiempo de inferencia para 1 símbolo.

        Returns
        -------
        etiquetas : np.ndarray int, shape (N,)
        """
        if not self._entrenado:
            raise RuntimeError(
                f"[{self.nombre}] El clasificador no ha sido entrenado. "
                "Llamar a fit() antes de predict()."
            )

        # Medir tiempo de inferencia unitaria (un solo símbolo)
        muestra_unica = X[0:1]
        t0 = time.perf_counter()
        _ = self._predict_interno(muestra_unica)
        self._tiempo_inferencia_1 = time.perf_counter() - t0

        # Predicción completa
        return self._predict_interno(X)

    # ------------------------------------------------------------------
    # Acceso a métricas de tiempo
    # ------------------------------------------------------------------

    @property
    def tiempo_entrenamiento(self) -> float:
        """Tiempo de entrenamiento en segundos (última llamada a fit)."""
        return self._tiempo_entrenamiento

    @property
    def tiempo_inferencia_unitaria(self) -> float:
        """Tiempo de inferencia para 1 símbolo en segundos."""
        return self._tiempo_inferencia_1

    def __repr__(self) -> str:
        estado = "entrenado" if self._entrenado else "no entrenado"
        return f"{self.__class__.__name__}({estado})"
