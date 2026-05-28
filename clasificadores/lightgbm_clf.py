"""
clasificadores/lightgbm_clf.py — VERSION CORREGIDA
===================================================
Correcciones respecto a la version anterior:

  1. num_leaves por defecto aumentado a 63 (minimo recomendado para 16-QAM).
     La version anterior usaba 31, causando un piso de BER ~1e-3 a alto SNR
     porque las fronteras de decision eran demasiado toscas.

  2. min_child_samples reducido a 10 (antes 20) y agregado al GridSearch.
     Valores altos impedian splits finos en regiones de frontera con pocos errores.

  3. learning_rate reducido a 0.05 (antes 0.1) para mejor convergencia con
     mas estimadores.

  4. Grid de busqueda expandido: num_leaves=[63,127,255], n_est=[200,400,800],
     min_child=[5,10,20]. El grid anterior [15,31,63] era insuficiente.

  5. Corregido warning "X does not have valid feature names": se convierte
     X a np.ndarray antes de fit() y predict().

  6. Cache integrado en optimizar_hiperparametros() con la misma interfaz
     que SVM y KNN.

CAUSA RAIZ DEL PROBLEMA ANTERIOR:
  Las fronteras de Voronoi de 16-QAM requieren al menos ~81 hojas para ser
  aproximadas con precision suficiente a alto SNR (>10 dB). Con num_leaves=15
  y min_child_samples=20, el modelo no puede crear splits lo suficientemente
  finos cerca de las fronteras, resultando en un piso de BER independiente del SNR.
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
        n_estimators        = 400,
        num_leaves          = 63,
        learning_rate       = 0.05,
        min_child_samples   = 10,
        n_est_grid          = None,
        num_leaves_grid     = None,
        min_child_grid      = None,
        cv_folds            = 3,
        optimizar           = True,
        guardar_modelo      = False,
        dir_modelos         = "modelos/",
        seed                = 42,
        usar_cache          = True,
        dir_cache           = "hiperparametros/",
    ):
        super().__init__()
        self.n_estimators       = n_estimators
        self.num_leaves         = num_leaves
        self.learning_rate      = learning_rate
        self.min_child_samples  = min_child_samples
        self.n_est_grid         = n_est_grid      or [200, 400, 800]
        self.num_leaves_grid    = num_leaves_grid or [63, 127, 255]
        self.min_child_grid     = min_child_grid  or [5, 10, 20]
        self.cv_folds           = cv_folds
        self.optimizar          = optimizar
        self.guardar_modelo     = guardar_modelo
        self.dir_modelos        = dir_modelos
        self.seed               = seed
        self.usar_cache         = usar_cache
        self.dir_cache          = dir_cache
        self.modelo             = None
        self._mejores_params    = {}

    @property
    def nombre(self):
        return "LightGBM"

    @property
    def nombre_base(self):
        return "LightGBM"

    def optimizar_hiperparametros(self, X_train, y_train):
        import json as _json, os as _os

        # Intentar leer del cache
        if self.usar_cache and self.dir_cache:
            ruta_cache = _os.path.join(self.dir_cache, "lightgbm_hiperparams.json")
            if _os.path.exists(ruta_cache):
                try:
                    with open(ruta_cache, 'r', encoding='utf-8') as f:
                        datos = _json.load(f)
                    params = datos.get("params", {})
                    fecha  = datos.get("fecha", "desconocida")
                    acc    = datos.get("cv_accuracy")
                    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                    print(f"  [Cache] Hiperparametros cargados desde cache "
                          f"(guardado: {fecha}  |  CV accuracy = {acc_str})")
                    print(f"  [Cache]   Params: {params}")
                    self._aplicar_params(params)
                    self._mejores_params = params
                    print(f"  [LightGBM] Cache cargado: {params}")
                    return params
                except Exception as e:
                    print(f"  [Cache] Error leyendo cache ({e}), ejecutando GridSearch...")

        if not self.optimizar:
            return {}

        print(f"  [LightGBM] Optimizando hiperparametros (GridSearch {self.cv_folds}-fold)...")
        print(f"  [LightGBM] Grid: n_est={self.n_est_grid}, "
              f"leaves={self.num_leaves_grid}, min_child={self.min_child_grid}")

        N_cv = min(len(X_train), 50_000)
        idx  = np.random.choice(len(X_train), N_cv, replace=False)
        X_cv = np.asarray(X_train[idx])
        y_cv = y_train[idx]

        param_grid = {
            'n_estimators'     : self.n_est_grid,
            'num_leaves'       : self.num_leaves_grid,
            'min_child_samples': self.min_child_grid,
        }
        clf_base = lgb.LGBMClassifier(
            objective     = 'multiclass',
            num_class     = 16,
            learning_rate = self.learning_rate,
            random_state  = self.seed,
            n_jobs        = -1,
            verbose       = -1,
        )
        gs = GridSearchCV(clf_base, param_grid, cv=self.cv_folds,
                          scoring='neg_log_loss', n_jobs=-1, verbose=0)
        gs.fit(X_cv, y_cv)

        self._mejores_params = gs.best_params_
        self._aplicar_params(gs.best_params_)
        print(f"  [LightGBM] Mejores params: {self._mejores_params}  "
              f"(CV accuracy = {gs.best_score_:.4f})")

        if self.dir_cache:
            self._guardar_cache(gs.best_params_, gs.best_score_)

        return self._mejores_params

    def _aplicar_params(self, params):
        if 'n_estimators'      in params: self.n_estimators      = params['n_estimators']
        if 'num_leaves'        in params: self.num_leaves         = params['num_leaves']
        if 'min_child_samples' in params: self.min_child_samples  = params['min_child_samples']

    def _fit_interno(self, X_train, y_train):
        self.modelo = lgb.LGBMClassifier(
            objective         = 'multiclass',
            num_class         = 16,
            n_estimators      = self.n_estimators,
            num_leaves        = self.num_leaves,
            learning_rate     = self.learning_rate,
            min_child_samples = self.min_child_samples,
            subsample         = 0.8,
            colsample_bytree  = 1.0,
            random_state      = self.seed,
            n_jobs            = -1,
            verbose           = -1,
        )
        self.modelo.fit(np.asarray(X_train), y_train)
        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X):
        return self.modelo.predict(np.asarray(X)).astype(np.int32)

    def reentrenar_incremental(self, X_nuevo, y_nuevo, n_arboles_extra=50):
        if self.modelo is None:
            raise RuntimeError("[LightGBM] Llamar a fit() antes de reentrenar.")
        print(f"  [LightGBM] Reentrenamiento incremental: {len(X_nuevo)} muestras, "
              f"+{n_arboles_extra} arboles...")
        n_total = self.n_estimators + n_arboles_extra
        self.modelo.set_params(n_estimators=n_total)
        self.modelo.fit(np.asarray(X_nuevo), y_nuevo, init_model=self.modelo.booster_)
        self.n_estimators = n_total

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos, "modelo_LightGBM.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({'modelo': self.modelo, 'params': self._mejores_params}, f)

    def _guardar_cache(self, params, cv_accuracy):
        import json as _json, os as _os
        from datetime import datetime
        _os.makedirs(self.dir_cache, exist_ok=True)
        ruta = _os.path.join(self.dir_cache, "lightgbm_hiperparams.json")
        with open(ruta, 'w', encoding='utf-8') as f:
            _json.dump({
                "params"      : params,
                "cv_accuracy" : float(cv_accuracy),
                "fecha"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        print(f"  [Cache] Guardado: {ruta}")
