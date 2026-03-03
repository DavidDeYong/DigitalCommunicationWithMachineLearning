"""
clasificadores/lightgbm_clf.py
==============================
Clasificador LightGBM para demodulacion 16-QAM.

LightGBM usa gradient boosting con crecimiento leaf-wise y construccion
de histogramas, 10-50x mas rapido que Random Forest con poder expresivo
comparable.

Sin supuestos distribucionales: aprende fronteras de decision arbitrarias,
por lo que se adapta a ruido no-Gaussiano, interferencia, o desplazamientos
de constelacion que el detector Bayesiano (optimizado para AWGN) no maneja.

Entrenamiento incremental:
  reentrenar_incremental() permite agregar arboles sobre el modelo existente
  sin partir de cero, usando los nuevos datos del canal modificado.

Instalacion: pip install lightgbm
"""

import numpy as np
import os
import pickle
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from clasificadores.base import ClasificadorBase


class ClasificadorLightGBM(ClasificadorBase):

    def __init__(
        self,
        n_estimators       = 200,
        num_leaves         = 31,
        learning_rate      = 0.1,
        min_child_samples  = 20,
        n_est_grid         = None,
        num_leaves_grid    = None,
        cv_folds           = 3,
        optimizar          = True,
        guardar_modelo     = False,
        dir_modelos        = "modelos/",
        seed               = 42,
    ):
        """
        Parameters
        ----------
        n_estimators      : numero de arboles (iteraciones de boosting)
        num_leaves        : maximo de hojas por arbol (controla complejidad)
        learning_rate     : tasa de aprendizaje del boosting
        min_child_samples : minimo de muestras por hoja (regularizacion)
        optimizar         : si True, hace GridSearch sobre n_estimators y num_leaves
        """
        super().__init__()
        self.n_estimators       = n_estimators
        self.num_leaves         = num_leaves
        self.learning_rate      = learning_rate
        self.min_child_samples  = min_child_samples
        self.n_est_grid         = n_est_grid     or [100, 200, 400]
        self.num_leaves_grid    = num_leaves_grid or [15, 31, 63]
        self.cv_folds           = cv_folds
        self.optimizar          = optimizar
        self.guardar_modelo     = guardar_modelo
        self.dir_modelos        = dir_modelos
        self.seed               = seed
        self.modelo             = None
        self._mejores_params    = {}

    @property
    def nombre(self):
        return "LightGBM"

    def optimizar_hiperparametros(self, X_train, y_train):
        if not self.optimizar:
            return {}

        print(f"  [LightGBM] Optimizando hiperparametros (GridSearch {self.cv_folds}-fold)...")

        # LightGBM es rapido, puede usar N mas grande que RF
        N_cv = min(len(X_train), 50_000)
        idx  = np.random.choice(len(X_train), N_cv, replace=False)
        X_cv, y_cv = X_train[idx], y_train[idx]

        param_grid = {
            'n_estimators': self.n_est_grid,
            'num_leaves':   self.num_leaves_grid,
        }
        clf_base = lgb.LGBMClassifier(
            objective         = 'multiclass',
            num_class         = 16,
            learning_rate     = self.learning_rate,
            min_child_samples = self.min_child_samples,
            random_state      = self.seed,
            n_jobs            = -1,
            verbose           = -1,
        )
        gs = GridSearchCV(clf_base, param_grid, cv=self.cv_folds,
                          scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_cv, y_cv)

        self._mejores_params = gs.best_params_
        self.n_estimators    = gs.best_params_['n_estimators']
        self.num_leaves      = gs.best_params_['num_leaves']
        print(f"  [LightGBM] Mejores params: {self._mejores_params}")
        return self._mejores_params

    def _fit_interno(self, X_train, y_train):
        self.modelo = lgb.LGBMClassifier(
            objective         = 'multiclass',
            num_class         = 16,
            n_estimators      = self.n_estimators,
            num_leaves        = self.num_leaves,
            learning_rate     = self.learning_rate,
            min_child_samples = self.min_child_samples,
            subsample         = 0.8,
            colsample_bytree  = 1.0,   # solo 2 features IQ, usar todas
            random_state      = self.seed,
            n_jobs            = -1,
            verbose           = -1,
        )
        self.modelo.fit(X_train, y_train)
        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X):
        return self.modelo.predict(X).astype(np.int32)

    def reentrenar_incremental(self, X_nuevo, y_nuevo, n_arboles_extra=50):
        """
        Agrega arboles al modelo existente entrenados sobre nuevos datos del canal.
        Los arboles previos se conservan; los nuevos corrigen los residuos en el
        nuevo escenario.

        Util cuando el canal cambia (ruido no-Gaussiano, interferencia, fading)
        y se dispone de una nueva trama piloto del escenario actualizado.

        Parameters
        ----------
        X_nuevo         : nuevos simbolos IQ (trama piloto del canal adaptado)
        y_nuevo         : etiquetas verdaderas correspondientes
        n_arboles_extra : arboles adicionales a agregar (default: 50)
        """
        if self.modelo is None:
            raise RuntimeError("[LightGBM] Llamar a fit() antes de reentrenar.")
        print(f"  [LightGBM] Reentrenamiento incremental: {len(X_nuevo)} muestras, "
              f"+{n_arboles_extra} arboles...")
        n_total = self.n_estimators + n_arboles_extra
        self.modelo.set_params(n_estimators=n_total)
        self.modelo.fit(X_nuevo, y_nuevo, init_model=self.modelo.booster_)
        self.n_estimators = n_total
        print(f"  [LightGBM] Modelo actualizado: {self.n_estimators} arboles totales.")

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos, "modelo_LightGBM.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({'modelo': self.modelo, 'params': self._mejores_params}, f)
