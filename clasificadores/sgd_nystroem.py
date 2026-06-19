"""
clasificadores/sgd_nystroem.py

Clasificador SGD con aproximacion de kernel Nystroem para demodulacion 16-QAM.
Combina Nystroem (mapeo RBF O(N*n_components)) con SGDClassifier (O(N)).
Sin supuestos distribucionales. Soporta partial_fit para adaptacion online.
"""

import numpy as np
import os
import pickle
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model         import SGDClassifier
from sklearn.preprocessing        import StandardScaler
from sklearn.pipeline             import Pipeline
from sklearn.model_selection      import GridSearchCV
from clasificadores.base          import ClasificadorBase
import hiperparametros_cache as cache


class ClasificadorSGDNystroem(ClasificadorBase):

    def __init__(
        self,
        n_components    = 300,
        gamma           = 0.5,
        alpha           = 1e-4,
        max_iter        = 50,
        tol             = 1e-4,
        n_iter_no_change= 5,
        n_comp_grid     = None,
        gamma_grid      = None,
        cv_folds        = 3,
        optimizar       = True,
        guardar_modelo  = False,
        dir_modelos     = "modelos/",
        seed            = 42,
        usar_cache      = True,
        dir_cache       = "hiperparametros/",
    ):
        super().__init__()
        self.n_components   = n_components
        self.gamma          = gamma
        self.alpha          = alpha
        self.max_iter       = max_iter
        self.tol            = tol
        self.n_iter_no_change = n_iter_no_change
        self.n_comp_grid    = n_comp_grid or [100, 300, 500]
        self.gamma_grid     = gamma_grid  or [0.1, 0.5, 1.0, 2.0]
        self.cv_folds       = cv_folds
        self.optimizar      = optimizar
        self.guardar_modelo = guardar_modelo
        self.dir_modelos    = dir_modelos
        self.seed           = seed
        self.usar_cache     = usar_cache
        self.dir_cache      = dir_cache
        self._scaler         = StandardScaler()
        self._nystroem       = None
        self._sgd            = None
        self._mejores_params = {}

    @property
    def nombre(self):
        return "SGD-Nystroem"

    def optimizar_hiperparametros(self, X_train, y_train):
        if not self.optimizar:
            return {}
        if self.usar_cache and cache.existe(self.nombre, self.dir_cache):
            params = cache.cargar(self.nombre, self.dir_cache)
            if params:
                self._mejores_params = params
                self.n_components    = params.get("n_components", self.n_components)
                self.gamma           = params.get("gamma",        self.gamma)
                print(f"  [SGD-Nystroem] Cache cargado: {params}")
                return self._mejores_params
        print(f"  [SGD-Nystroem] Optimizando hiperparametros (GridSearch {self.cv_folds}-fold)...")
        N_cv = min(len(X_train), 30_000)
        idx  = np.random.choice(len(X_train), N_cv, replace=False)
        X_cv, y_cv = X_train[idx], y_train[idx]
        pipe = Pipeline([
            ("scaler",   StandardScaler()),
            ("nystroem", Nystroem(kernel="rbf", random_state=self.seed)),
            ("sgd",      SGDClassifier(
                loss="modified_huber", alpha=self.alpha,
                max_iter=self.max_iter, random_state=self.seed, n_jobs=-1,
            )),
        ])
        param_grid = {
            "nystroem__n_components": self.n_comp_grid,
            "nystroem__gamma":        self.gamma_grid,
        }
        gs = GridSearchCV(pipe, param_grid, cv=self.cv_folds,
                          scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(X_cv, y_cv)
        self._mejores_params = {
            "n_components": gs.best_params_["nystroem__n_components"],
            "gamma":        gs.best_params_["nystroem__gamma"],
        }
        self.n_components = self._mejores_params["n_components"]
        self.gamma        = self._mejores_params["gamma"]
        print(f"  [SGD-Nystroem] Mejores params: {self._mejores_params}")
        cache.guardar(self.nombre, self._mejores_params, self.dir_cache,
                      cv_accuracy=gs.best_score_)
        return self._mejores_params

    def _fit_interno(self, X_train, y_train):
        X_sc = self._scaler.fit_transform(X_train)
        self._nystroem = Nystroem(
            kernel="rbf", gamma=self.gamma,
            n_components=self.n_components, random_state=self.seed,
        )
        X_ny = self._nystroem.fit_transform(X_sc)
        self._sgd = SGDClassifier(
            loss="modified_huber", alpha=self.alpha,
            max_iter=self.max_iter, tol=self.tol,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.seed, n_jobs=-1,
        )
        self._sgd.fit(X_ny, y_train)

        n_iter = int(np.max(self._sgd.n_iter_))
        if n_iter >= self.max_iter:
            print(f"  [SGD-Nystroem] ⚠ Alcanzó max_iter={self.max_iter} sin converger "
                  f"(tol={self.tol}).")
        else:
            print(f"  [SGD-Nystroem] Convergió/detuvo en época {n_iter} "
                  f"(tol={self.tol}, n_iter_no_change={self.n_iter_no_change}).")

        # ── FLOPs analíticos (estimado) ──────────────────────────────────
        # Inferencia por muestra:
        #   1) Mapeo Nystroem: kernel RBF contra L landmarks → L×(3d + ~10) ops
        #      (resta, cuadrado, suma por dimensión + exponencial)
        #   2) Proyección al subespacio: L×L MACs (multiplicación por normalization_)
        #   3) Capa lineal SGD: L×C MACs
        L = self.n_components
        d = X_train.shape[1]
        C_cls = len(self._sgd.classes_)
        flops_kernel = L * (3 * d + 10)
        flops_proy   = 2 * L * L
        flops_lineal = 2 * L * C_cls
        self.flops_inferencia = float(flops_kernel + flops_proy + flops_lineal)
        self.n_parametros     = int(L * d + L * L + (L + 1) * C_cls)
        self.flops_tipo       = "estimado"
        # Entrenamiento: mapeado Nystroem (N muestras) + SGD pase por los datos
        N = X_train.shape[0]
        # Nystroem sobre N muestras: N * (kernel_per_landmark + proyeccion) ≈ N * L * 3d
        flops_ny_train = N * L * 3 * d
        # SGD: n_iter pases completos con actualización lineal (L → C)
        flops_sgd_train = 2 * N * L * C_cls * n_iter
        self.flops_entrenamiento = float(flops_ny_train + flops_sgd_train)

        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X):
        X_sc = self._scaler.transform(X)
        X_ny = self._nystroem.transform(X_sc)
        return self._sgd.predict(X_ny).astype(np.int32)

    def reentrenar_incremental(self, X_nuevo, y_nuevo, n_epochs=5):
        """
        Adapta el modelo online a nuevas condiciones del canal via partial_fit.
        La transformacion Nystroem se mantiene fija; solo se actualizan pesos SGD.
        """
        if self._sgd is None:
            raise RuntimeError("[SGD-Nystroem] Llamar a fit() antes de reentrenar.")
        X_sc   = self._scaler.transform(X_nuevo)
        X_ny   = self._nystroem.transform(X_sc)
        clases = np.arange(16)
        for _ in range(n_epochs):
            idx = np.random.permutation(len(X_ny))
            self._sgd.partial_fit(X_ny[idx], y_nuevo[idx], classes=clases)
        print(f"  [SGD-Nystroem] Adaptacion completada ({len(X_nuevo)} muestras, {n_epochs} epocas).")

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos, "modelo_SGD_Nystroem.pkl")
        with open(ruta, "wb") as f:
            pickle.dump({
                "scaler": self._scaler, "nystroem": self._nystroem,
                "sgd": self._sgd, "params": self._mejores_params,
            }, f)
