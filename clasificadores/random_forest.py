"""
clasificadores/random_forest.py
================================
Clasificador Random Forest para demodulación 16-QAM.
"""

import numpy as np
import os
import pickle
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .base import ClasificadorBase
import hiperparametros_cache as cache


class ClasificadorRandomForest(ClasificadorBase):

    def __init__(
        self,
        n_estimators:   int   = 100,
        max_depth:      int   = 8,
        n_trees_grid:   list  = None,
        max_depth_grid: list  = None,
        cv_folds:       int   = 3,
        optimizar:      bool  = True,
        guardar_modelo: bool  = False,
        dir_modelos:    str   = "modelos/",
        seed:           int   = 42,
        usar_cache:     bool  = True,
        dir_cache:      str   = "hiperparametros/",
    ):
        super().__init__()
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.n_trees_grid   = n_trees_grid   or [50, 100, 200]
        self.max_depth_grid = max_depth_grid or [6, 8, 12, None]
        self.cv_folds       = cv_folds
        self.optimizar      = optimizar
        self.guardar_modelo = guardar_modelo
        self.dir_modelos    = dir_modelos
        self.seed           = seed
        self.usar_cache     = usar_cache
        self.dir_cache      = dir_cache

        self.modelo          = None
        self._mejores_params: dict = {}

    @property
    def nombre(self) -> str:
        return "Random Forest"

    def optimizar_hiperparametros(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> dict:
        if not self.optimizar:
            return {}

        # Intentar cargar desde cache
        if self.usar_cache and cache.existe(self.nombre, self.dir_cache):
            params = cache.cargar(self.nombre, self.dir_cache)
            if params:
                self._mejores_params = params
                self.n_estimators = params.get('n_estimators', self.n_estimators)
                self.max_depth    = params.get('max_depth',    self.max_depth)
                return self._mejores_params

        print(f"  [RF] Optimizando hiperparámetros (GridSearch {self.cv_folds}-fold)...")

        # Submuestrear para acelerar GridSearch
        N_cv = min(len(X_train), 30_000)
        idx  = np.random.choice(len(X_train), N_cv, replace=False)
        X_cv = X_train[idx]
        y_cv = y_train[idx]

        param_grid = {
            'n_estimators': self.n_trees_grid,
            'max_depth':    self.max_depth_grid,
        }
        rf = RandomForestClassifier(
            criterion='gini', max_features='sqrt',
            bootstrap=True, n_jobs=-1, random_state=self.seed
        )
        gs = GridSearchCV(rf, param_grid, cv=self.cv_folds,
                          scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_cv, y_cv)

        self._mejores_params = gs.best_params_
        self.n_estimators    = gs.best_params_['n_estimators']
        self.max_depth       = gs.best_params_['max_depth']

        print(f"  [RF] Mejores params: {self._mejores_params}")

        # Guardar en cache
        cache.guardar(self.nombre, self._mejores_params, self.dir_cache,
                      cv_accuracy=gs.best_score_)
        return self._mejores_params

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.modelo = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion='gini',
            max_depth=self.max_depth,
            min_samples_split=4,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=self.seed,
        )
        self.modelo.fit(X_train, y_train)

        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.predict(X).astype(np.int32)

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos,
                            f"modelo_RF_{self.n_estimators}trees.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({'modelo': self.modelo, 'params': self._mejores_params}, f)
