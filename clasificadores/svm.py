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

# --- ThunderSVM (GPU-accelerated SVM) ---
# Requires the native library to be compiled. If unavailable or not built,
# fall back silently to scikit-learn SVC (CPU).
_THUNDERSVM_AVAILABLE = False
ThunderSVC = None
try:
    from thundersvm import SVC as ThunderSVC
    _THUNDERSVM_AVAILABLE = True
except Exception:
    pass  # ThunderSVM not available; will use sklearn SVC

# scikit-learn is always available as the CPU baseline
from sklearn.svm             import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import StandardScaler
from .base import ClasificadorBase
import hiperparametros_cache as cache
from config import USE_GPU_SVM

# Module-level flag for easy introspection from correr_lote.py
THUNDERSVM_AVAILABLE = _THUNDERSVM_AVAILABLE


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
        max_iter:           int  = -1,      # -1 = sin límite (comportamiento sklearn por defecto)
        tol:                float = 1e-4,   # tolerancia de convergencia
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
        self.max_iter       = max_iter
        self.tol            = tol

        self.scaler = StandardScaler()
        self.modelo = None
        self._mejores_params: dict = {}


    def _construir_modelo(self):
        """Construye SVC con los parámetros actuales."""
        kwargs = dict(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma if self.kernel != 'linear' else 'scale',
            decision_function_shape='ovo',
            cache_size=500,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        if USE_GPU_SVM and THUNDERSVM_AVAILABLE and ThunderSVC is not None:
            return ThunderSVC(**kwargs)
        return SVC(**kwargs)


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

        svc = SVC(kernel=self.kernel, decision_function_shape='ovo', cache_size=500,
                  max_iter=self.max_iter, tol=self.tol)
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

        # ── FLOPs analíticos SVM (OVO) ──────────────────────────────────
        # Inferencia: para cada uno de los C(16,2)=120 clasificadores binarios,
        # se evalúa el kernel sobre los SVs de ese clasificador.
        # Kernel RBF: 2*d operaciones por SV (distancia euclidiana)
        # Kernel Lineal: 2*d operaciones por SV (producto punto)
        try:
            n_sv_total = self.modelo.support_vectors_.shape[0]
            d          = X_train.shape[1]
            n_pares    = 120   # C(16,2) clasificadores OVO
            # Aprox: distribución uniforme de SVs entre clasificadores
            sv_por_par = n_sv_total / n_pares
            self.flops_inferencia = float(2 * d * sv_por_par * n_pares)
            self.n_parametros     = int(n_sv_total)
            self.flops_tipo       = "estimado"
            # Entrenamiento SVM: costo dominante = evaluaciones de kernel en SMO
            # Aprox: n_sv * N * 2d (kernel entre SVs y muestras de entrenamiento)
            N = X_train.shape[0]
            self.flops_entrenamiento = float(n_sv_total * N * 2 * d)
        except Exception as e:
            print(f"  [{self.nombre}] [WARN] No se pudo estimar FLOPs: {e}")

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
        max_iter:       int   = -1,
        tol:            float = 1e-4,
    ):
        super().__init__(
            kernel='rbf', C=C, gamma=gamma,
            C_grid=C_grid or [0.1, 1, 10, 100],
            gamma_grid=gamma_grid or [0.01, 0.1, 1, 'scale'],
            cv_folds=cv_folds, optimizar=optimizar,
            guardar_modelo=guardar_modelo, dir_modelos=dir_modelos,
            usar_cache=usar_cache, dir_cache=dir_cache,
            max_iter=max_iter, tol=tol,
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
        max_iter:       int   = -1,
        tol:            float = 1e-4,
    ):
        super().__init__(
            kernel='linear', C=C, gamma='scale',
            C_grid=C_grid or [0.1, 1, 10, 100],
            gamma_grid=[],
            cv_folds=cv_folds, optimizar=optimizar,
            guardar_modelo=guardar_modelo, dir_modelos=dir_modelos,
            usar_cache=usar_cache, dir_cache=dir_cache,
            max_iter=max_iter, tol=tol,
        )

    @property
    def nombre(self) -> str:
        return "SVM Lineal"
