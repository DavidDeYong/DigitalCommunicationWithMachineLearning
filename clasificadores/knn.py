"""
clasificadores/knn.py
=====================
Clasificador K-Nearest Neighbors (KNN) para demodulación 16-QAM.

La búsqueda de K óptimo se realiza por validación cruzada sobre el conjunto
de entrenamiento, evaluando todos los K del grid definido en config.py.
"""

import numpy as np
import os
import pickle
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from .base import ClasificadorBase
import hiperparametros_cache as cache


class ClasificadorKNN(ClasificadorBase):

    def __init__(
        self,
        k:              int   = 15,
        k_grid:         list  = None,
        cv_folds:       int   = 5,
        optimizar:      bool  = True,
        guardar_modelo: bool  = False,
        dir_modelos:    str   = "modelos/",
        usar_cache:     bool  = True,
        dir_cache:      str   = "hiperparametros/",
    ):
        """
        Parameters
        ----------
        k           : valor de K por defecto (si optimizar=False)
        k_grid      : valores de K a evaluar en la búsqueda
        cv_folds    : pliegues para validación cruzada
        optimizar   : si True, busca K óptimo automáticamente
        usar_cache  : si True, carga K óptimo desde cache si existe
        dir_cache   : directorio donde se guarda/lee el cache JSON
        """
        super().__init__()
        self.k              = k
        self.k_grid         = k_grid or [3, 7, 15, 25, 51]
        self.cv_folds       = cv_folds
        self.optimizar      = optimizar
        self.guardar_modelo = guardar_modelo
        self.dir_modelos    = dir_modelos
        self.usar_cache     = usar_cache
        self.dir_cache      = dir_cache

        self.modelo          = None
        self._mejores_params: dict = {}

    @property
    def nombre(self) -> str:
        return f"KNN (k={self.k})"

    def optimizar_hiperparametros(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> dict:
        """
        Busca el K óptimo por validación cruzada.
        Si existe cache, lo usa directamente sin buscar.
        """
        if not self.optimizar:
            return {}

        # Intentar cargar desde cache
        if self.usar_cache and cache.existe(self.nombre_base, self.dir_cache):
            params = cache.cargar(self.nombre_base, self.dir_cache)
            if params:
                self._mejores_params = params
                self.k = params.get('k', self.k)
                return self._mejores_params

        print(f"  [KNN] Optimizando K en {self.k_grid} con CV={self.cv_folds}...")

        # Submuestrear para acelerar CV si el dataset es muy grande
        N_cv = min(len(X_train), 20_000)
        idx  = np.random.choice(len(X_train), N_cv, replace=False)
        X_cv = X_train[idx]
        y_cv = y_train[idx]

        mejores_k     = self.k
        mejor_score   = -1.0

        for k in self.k_grid:
            knn    = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            scores = cross_val_score(knn, X_cv, y_cv, cv=self.cv_folds,
                                     scoring='accuracy', n_jobs=-1)
            media  = scores.mean()
            print(f"    K={k:3d}  →  accuracy CV = {media:.4f} ± {scores.std():.4f}")

            if media > mejor_score:
                mejor_score = media
                mejores_k   = k

        self.k               = mejores_k
        self._mejores_params = {'k': mejores_k, 'cv_accuracy': mejor_score}
        print(f"  [KNN] K óptimo = {mejores_k}  (accuracy CV = {mejor_score:.4f})")

        # Guardar en cache
        cache.guardar(self.nombre_base, self._mejores_params, self.dir_cache,
                      cv_accuracy=mejor_score)
        return self._mejores_params

    @property
    def nombre_base(self) -> str:
        """Nombre para el cache (sin el K actual, que puede cambiar)."""
        return "KNN" 

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.modelo = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        self.modelo.fit(X_train, y_train)

        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.predict(X).astype(np.int32)

    def _calcular_flops(self) -> int:
        # Distancia euclidiana a cada punto de entrenamiento: d=2 → 2 restas + 2 cuadrados + 1 suma = 5
        n_train = len(self.modelo._fit_X)
        flops_distancias = n_train * 5
        # Ordenamiento parcial para los k vecinos: ~n_train comparaciones
        flops_sort = n_train
        # Voto mayoritario entre k vecinos
        flops_voto = self.k - 1
        return flops_distancias + flops_sort + flops_voto

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos, f"modelo_KNN_k{self.k}.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({'modelo': self.modelo, 'params': self._mejores_params}, f)
