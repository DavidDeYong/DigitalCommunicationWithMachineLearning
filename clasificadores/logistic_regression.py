"""
clasificadores/logistic_regression.py
======================================
Clasificador de Regresión Logística multinomial para el benchmark 16-QAM.

Características:
  - Clasificación multinomial directa (16 clases) con softmax
  - Solver 'lbfgs' (recomendado para multiclase, convergencia rápida)
  - Regularización L2 con hiperparámetro C (inverso de la fuerza de regularización)
  - GridSearch con cache de hiperparámetros, igual que SVM/KNN/RF
  - Entrenamiento muy rápido: una sola optimización convexa, sin iteraciones SGD

Posición en el benchmark:
  - Más rápido que SVM para grandes N (complejidad O(N·d) vs O(N²-N³))
  - Fronteras de decisión lineales por par de clases → comparable a SVM Lineal
  - Útil como baseline rápido con garantías teóricas de convergencia

Parámetros relevantes:
  C           : inverso de regularización (mayor C = menos regularización)
  max_iter    : iteraciones máximas del solver (aumentar si no converge)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import hiperparametros_cache as cache

from .base import ClasificadorBase


class ClasificadorLogisticRegression(ClasificadorBase):
    """
    Regresión Logística multinomial para demodulación 16-QAM.

    Parámetros
    ----------
    C              : fuerza de regularización inversa (default: 1.0)
    C_grid         : valores de C para GridSearch
    cv_folds       : número de folds para validación cruzada
    max_iter       : máximo de iteraciones del solver lbfgs
    optimizar      : si True, ejecuta GridSearch en lugar de usar defaults
    usar_cache     : si True, carga/guarda hiperparámetros desde disco
    dir_cache      : carpeta del cache de hiperparámetros
    guardar_modelo : reservado para compatibilidad (no usado en sklearn)
    dir_modelos    : reservado para compatibilidad
    """

    def __init__(
        self,
        C: float             = 1.0,
        C_grid: list         = None,
        cv_folds: int        = 3,
        max_iter: int        = 1000,
        optimizar: bool      = True,
        usar_cache: bool     = True,
        dir_cache: str       = "hiperparametros/",
        guardar_modelo: bool = False,
        dir_modelos: str     = "modelos/",
    ):
        super().__init__()
        self._C        = C
        self._C_grid   = C_grid or [0.01, 0.1, 1.0, 10.0, 100.0]
        self._cv_folds = cv_folds
        self._max_iter = max_iter
        self._optimizar  = optimizar
        self._usar_cache = usar_cache
        self._dir_cache  = dir_cache
        self._modelo     = None

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    @property
    def nombre(self) -> str:
        return "Logistic Reg."

    # ------------------------------------------------------------------
    # Hiperparámetros — GridSearch con cache
    # ------------------------------------------------------------------

    def optimizar_hiperparametros(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict:
        """
        Busca el mejor C por validación cruzada (scoring: accuracy).
        Carga desde cache si está disponible y usar_cache=True.
        """
        CACHE_KEY = "logistic_regression"

        # ── Intentar cargar desde cache ──────────────────────────────
        if self._usar_cache:
            params = cache.cargar(CACHE_KEY, self._dir_cache)
            if params:
                self._C = float(params.get("C", self._C))
                print(f"  [Cache] ✓ Hiperparámetros LR cargados "
                      f"(guardado: {params.get('fecha', '?')}  |  "
                      f"CV accuracy = {params.get('cv_accuracy', '?')})")
                print(f"  [Cache]   Params: {{'C': {self._C}}}")
                return {"C": self._C}

        # ── GridSearch manual ────────────────────────────────────────
        if not self._optimizar:
            return {"C": self._C}

        # Subsampleo para acelerar GridSearch (máx 50k símbolos)
        N_MAX_GS = 50_000
        if len(X_train) > N_MAX_GS:
            idx = np.random.choice(len(X_train), N_MAX_GS, replace=False)
            Xgs, ygs = X_train[idx], y_train[idx]
        else:
            Xgs, ygs = X_train, y_train

        print(f"  [LR] GridSearch sobre C={self._C_grid} "
              f"({self._cv_folds}-fold CV, N={len(Xgs):,})")

        mejor_C, mejor_acc = self._C, -1.0

        for C_val in self._C_grid:
            modelo = LogisticRegression(
                C=C_val,
                solver="lbfgs",
                max_iter=self._max_iter,
                random_state=42,
            )
            scores = cross_val_score(modelo, Xgs, ygs,
                                     cv=self._cv_folds,
                                     scoring="accuracy",
                                     n_jobs=-1)
            acc = scores.mean()
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_C   = C_val

        self._C = mejor_C
        print(f"  [LR] Mejor: C={mejor_C}  CV accuracy={mejor_acc:.4f}")

        # ── Guardar en cache ─────────────────────────────────────────
        if self._usar_cache:
            cache.guardar(CACHE_KEY,
                          {"C": mejor_C},
                          self._dir_cache,
                          cv_accuracy=mejor_acc)

        return {"C": self._C}

    # ------------------------------------------------------------------
    # Entrenamiento y predicción
    # ------------------------------------------------------------------

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._modelo = LogisticRegression(
            C=self._C,
            solver="lbfgs",
            max_iter=self._max_iter,
            random_state=42,
        )
        self._modelo.fit(X_train, y_train)

    def _calcular_flops(self) -> int:
        # coef_ shape: (16, 2)  →  X @ coef_.T: 16 × 2 × 2 MACs = 64, + 16 bias = 80
        # softmax: 16 exp + 15 sumas + 16 divisiones = 47
        # argmax: 15 comparaciones
        # StandardScaler implicito no se usa aqui (LR opera sobre datos crudos)
        return 80 + 47 + 15  # = 142

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        return self._modelo.predict(X)
