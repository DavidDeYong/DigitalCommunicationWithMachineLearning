"""
clasificadores/svm.py
=====================
Clasificadores basados en Support Vector Machine (SVM) para demodulación 16-QAM.

Se implementan dos variantes:
  - SVM con kernel RBF    (ClasificadorSVM_RBF)
  - SVM con kernel Lineal (ClasificadorSVM_Lineal)

Ambas heredan de _ClasificadorSVM_Base que concentra la lógica común
(GridSearch, interfaz, guardado de modelo).

Estrategia multiclase: OVO (one-vs-one), que es el default de sklearn SVC.
Con 16 clases genera C(16,2)=120 clasificadores binarios.
"""

import numpy as np
import os
import pickle
from sklearn.svm            import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing  import StandardScaler
from .base import ClasificadorBase
import hiperparametros_cache as cache


class _ClasificadorSVM_Base(ClasificadorBase):
    """Clase base interna para ambas variantes SVM."""

    def __init__(
        self,
        kernel:             str,
        C:                  float,
        gamma,
        C_grid:             list,
        gamma_grid:         list,
        cv_folds:           int,
        optimizar:          bool,
        guardar_modelo:     bool,
        dir_modelos:        str,
        usar_cache:         bool = True,
        dir_cache:          str  = "hiperparametros/",
    ):
        super().__init__()
        self.kernel         = kernel
        self.C              = C
        self.gamma          = gamma
        self.C_grid         = C_grid
        self.gamma_grid     = gamma_grid
        self.cv_folds       = cv_folds
        self.optimizar      = optimizar
        self.guardar_modelo = guardar_modelo
        self.dir_modelos    = dir_modelos
        self.usar_cache     = usar_cache
        self.dir_cache      = dir_cache

        self.scaler = StandardScaler()
        self.modelo = None
        self._mejores_params: dict = {}

    def _construir_modelo(self):
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma if self.kernel != 'linear' else 'scale',
            decision_function_shape='ovo',
            cache_size=500,
        )

    def optimizar_hiperparametros(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> dict:
        """GridSearch sobre C y gamma. Si existe cache, lo usa en lugar de buscar."""
        if not self.optimizar:
            return {}

        # Intentar cargar desde cache
        if self.usar_cache and cache.existe(self.nombre, self.dir_cache):
            params = cache.cargar(self.nombre, self.dir_cache)
            if params:
                self._mejores_params = params
                self.C     = params.get('C', self.C)
                if self.kernel != 'linear':
                    self.gamma = params.get('gamma', self.gamma)
                return self._mejores_params

        # Ejecutar GridSearch
        print(f"  [{self.nombre}] Optimizando hiperparámetros (GridSearch {self.cv_folds}-fold)...")

        X_sc = self.scaler.fit_transform(X_train)

        if self.kernel == 'linear':
            param_grid = {'C': self.C_grid}
        else:
            param_grid = {'C': self.C_grid, 'gamma': self.gamma_grid}

        svc = SVC(kernel=self.kernel, decision_function_shape='ovo', cache_size=500)
        gs  = GridSearchCV(svc, param_grid, cv=self.cv_folds, scoring='accuracy',
                           n_jobs=-1, verbose=0)
        gs.fit(X_sc, y_train)

        self._mejores_params = gs.best_params_
        self.C     = gs.best_params_['C']
        if self.kernel != 'linear':
            self.gamma = gs.best_params_.get('gamma', self.gamma)

        print(f"  [{self.nombre}] Mejores params: {self._mejores_params}")

        # Guardar en cache para futuras corridas
        cache.guardar(self.nombre, self._mejores_params, self.dir_cache,
                      cv_accuracy=gs.best_score_)
        return self._mejores_params

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_sc        = self.scaler.fit_transform(X_train)
        self.modelo = self._construir_modelo()
        self.modelo.fit(X_sc, y_train)

        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler.transform(X)
        return self.modelo.predict(X_sc).astype(np.int32)

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        nombre_archivo = os.path.join(
            self.dir_modelos, f"modelo_{self.nombre.replace(' ', '_')}.pkl"
        )
        with open(nombre_archivo, 'wb') as f:
            pickle.dump({'modelo': self.modelo, 'scaler': self.scaler,
                         'params': self._mejores_params}, f)


# ---------------------------------------------------------------------------
# Variante RBF
# ---------------------------------------------------------------------------

class ClasificadorSVM_RBF(_ClasificadorSVM_Base):

    def __init__(
        self,
        C:              float = 10.0,
        gamma                 = 0.1,
        C_grid:         list  = None,
        gamma_grid:     list  = None,
        cv_folds:       int   = 5,
        optimizar:      bool  = True,
        guardar_modelo: bool  = False,
        dir_modelos:    str   = "modelos/",
        usar_cache:     bool  = True,
        dir_cache:      str   = "hiperparametros/",
    ):
        super().__init__(
            kernel='rbf', C=C, gamma=gamma,
            C_grid=C_grid or [0.1, 1, 10, 100],
            gamma_grid=gamma_grid or [0.01, 0.1, 1, 'scale'],
            cv_folds=cv_folds, optimizar=optimizar,
            guardar_modelo=guardar_modelo, dir_modelos=dir_modelos,
            usar_cache=usar_cache, dir_cache=dir_cache,
        )

    @property
    def nombre(self) -> str:
        return "SVM RBF"


# ---------------------------------------------------------------------------
# Variante Lineal
# ---------------------------------------------------------------------------

class ClasificadorSVM_Lineal(_ClasificadorSVM_Base):

    def __init__(
        self,
        C:              float = 10.0,
        C_grid:         list  = None,
        cv_folds:       int   = 5,
        optimizar:      bool  = True,
        guardar_modelo: bool  = False,
        dir_modelos:    str   = "modelos/",
        usar_cache:     bool  = True,
        dir_cache:      str   = "hiperparametros/",
    ):
        super().__init__(
            kernel='linear', C=C, gamma='scale',
            C_grid=C_grid or [0.1, 1, 10, 100],
            gamma_grid=[],
            cv_folds=cv_folds, optimizar=optimizar,
            guardar_modelo=guardar_modelo, dir_modelos=dir_modelos,
            usar_cache=usar_cache, dir_cache=dir_cache,
        )

    @property
    def nombre(self) -> str:
        return "SVM Lineal"
