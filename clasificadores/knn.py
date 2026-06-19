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
        max_train_samples: int = None,
        seed:           int   = 42,
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
        max_train_samples : si no es None y N_train lo supera, se aplica
                      submuestreo ESTRATIFICADO por clase (instance selection).
                      KNN es lazy: almacenar todo N_train es redundante cuando
                      la accuracy ya saturó. Reduce memoria y operaciones.
        seed        : semilla para el submuestreo (reproducibilidad)
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
        self.max_train_samples = max_train_samples
        self.seed           = seed

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

    def _submuestrear_estratificado(self, X, y):
        """
        Submuestreo estratificado por clase a max_train_samples muestras.
        Mantiene las proporciones de clase originales (con piloto equiprobable,
        ~max_train_samples/16 por símbolo).
        """
        rng  = np.random.default_rng(self.seed)
        frac = self.max_train_samples / len(X)
        idx_sel = []
        for c in np.unique(y):
            idx_c = np.flatnonzero(y == c)
            n_c   = max(1, int(round(len(idx_c) * frac)))
            idx_sel.append(rng.choice(idx_c, size=n_c, replace=False))
        idx_sel = np.concatenate(idx_sel)
        print(f"  [KNN] Submuestreo estratificado: {len(X):,} → {len(idx_sel):,} "
              f"muestras almacenadas")
        return X[idx_sel], y[idx_sel]

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if (self.max_train_samples is not None
                and len(X_train) > self.max_train_samples):
            X_train, y_train = self._submuestrear_estratificado(X_train, y_train)

        self.modelo = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        self.modelo.fit(X_train, y_train)

        # ── FLOPs analíticos (estimado, cota de fuerza bruta) ────────────
        # Inferencia: distancia euclidiana a las N_train muestras almacenadas
        # → 3 ops por dimensión (resta, cuadrado, suma) por muestra.
        # Nota: sklearn usa KD-tree/Ball-tree, que reduce el costo medio,
        # pero la cota N×3d es la referencia estándar del algoritmo.
        N, d = X_train.shape
        self.flops_inferencia    = float(N * 3 * d)
        self.n_parametros        = int(N * d)   # muestras almacenadas
        self.flops_tipo          = "estimado"
        # KNN no tiene entrenamiento real: solo copia datos en memoria
        self.flops_entrenamiento = float(N * d)

        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.predict(X).astype(np.int32)

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos, f"modelo_KNN_k{self.k}.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({'modelo': self.modelo, 'params': self._mejores_params}, f)
