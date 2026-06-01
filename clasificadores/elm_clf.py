"""
clasificadores/elm_clf.py
=========================
Clasificador ELM con cuatro estrategias de inicialización de pesos.

Estrategias
-----------
'aleatorio'    : W ~ N(0,1), b ~ N(0,1)  [comportamiento original]
'constelacion' : cada neurona centrada cerca de un símbolo 16-QAM
'voronoi'      : neuronas orientadas a los vectores normales de las fronteras
                 de decisión (I=0,±2 y Q=0,±2)
'hibrido'      : 50% constelacion + 50% aleatorio
"""

import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from clasificadores.base import ClasificadorBase

try:
    import torch
    _TORCH_DISPONIBLE = True
except ImportError:
    _TORCH_DISPONIBLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Geometría de la constelación 16-QAM
# ─────────────────────────────────────────────────────────────────────────────

_CONSTELACION_16QAM = np.array([
    [-3,-3],[-3,-1],[-3, 1],[-3, 3],
    [-1,-3],[-1,-1],[-1, 1],[-1, 3],
    [ 1,-3],[ 1,-1],[ 1, 1],[ 1, 3],
    [ 3,-3],[ 3,-1],[ 3, 1],[ 3, 3],
], dtype=np.float64)

# Vectores normales a las 8 fronteras de Voronoi (I=0,±2  y  Q=0,±2)
_FRONTERAS_W = np.array([
    [ 1, 0],[-1, 0],[ 1, 0],[-1, 0],
    [ 0, 1],[ 0,-1],[ 0, 1],[ 0,-1],
], dtype=np.float64)
_FRONTERAS_B = np.array([0, 0, -2, 2, 0, 0, -2, 2], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _activacion(H_lin, func):
    if func == 'relu':    return np.maximum(0, H_lin)
    if func == 'sigmoid': return 1.0 / (1.0 + np.exp(-np.clip(H_lin,-500,500)))
    if func == 'tanh':    return np.tanh(H_lin)
    raise ValueError(f"activation desconocida: '{func}'")

def _one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ─────────────────────────────────────────────────────────────────────────────
# Núcleo ELM
# ─────────────────────────────────────────────────────────────────────────────

class _ELMCore:
    def __init__(self, hidden_units, activation_function, C,
                 random_type='normal', seed=42,
                 init_strategy='aleatorio', sigma_init=0.5):
        self.L             = hidden_units
        self.act           = activation_function
        self.C             = C
        self.rtype         = random_type
        self.seed          = seed
        self.init_strategy = init_strategy
        self.sigma_init    = sigma_init
        self.W = self.b = self.beta = self.n_classes = None

    # ── Estrategias de inicialización ─────────────────────────────────────

    def _init_aleatorio(self, n_features, rng):
        if self.rtype == 'normal':
            self.W = rng.standard_normal((self.L, n_features))
            self.b = rng.standard_normal(self.L)
        else:
            self.W = rng.uniform(-1, 1, (self.L, n_features))
            self.b = rng.uniform(-1, 1, self.L)

    def _init_constelacion(self, rng):
        """
        Cada neurona se centra cerca de un símbolo de la constelación.
        Con L neuronas → cada símbolo recibe L//16 neuronas (+ reparto
        del resto). Garantiza cobertura uniforme del espacio I-Q.
        """
        idx    = np.arange(self.L) % 16
        W_base = _CONSTELACION_16QAM[idx]
        self.W = W_base + rng.normal(0, self.sigma_init, W_base.shape)
        self.b = rng.normal(0, self.sigma_init, self.L)

    def _init_voronoi(self, rng):
        """
        Neuronas orientadas a los vectores normales de las fronteras de
        Voronoi de la constelación 16-QAM rectangular.
        Cada neurona actúa como detector de un semiplano de decisión.
        """
        idx    = np.arange(self.L) % len(_FRONTERAS_W)
        self.W = _FRONTERAS_W[idx].copy()
        self.b = _FRONTERAS_B[idx].copy()
        # Pequeña perturbación para romper la simetría exacta
        self.W += rng.normal(0, 0.05, self.W.shape)
        self.b += rng.normal(0, 0.05, self.b.shape)

    def _init_hibrido(self, rng):
        """
        50% neuronas centradas en la constelación + 50% aleatorias.
        Preserva cobertura geométrica con flexibilidad para canales variables.
        """
        L_geo  = self.L // 2
        L_rnd  = self.L - L_geo
        idx    = np.arange(L_geo) % 16
        W_geo  = _CONSTELACION_16QAM[idx] + rng.normal(0, self.sigma_init, (L_geo, 2))
        b_geo  = rng.normal(0, self.sigma_init, L_geo)
        W_rnd  = rng.standard_normal((L_rnd, 2))
        b_rnd  = rng.standard_normal(L_rnd)
        self.W = np.vstack([W_geo, W_rnd])
        self.b = np.concatenate([b_geo, b_rnd])

    def _init_pesos(self, n_features, rng):
        dispatch = {
            'aleatorio'   : self._init_aleatorio,
            'constelacion': self._init_constelacion,
            'voronoi'     : self._init_voronoi,
            'hibrido'     : self._init_hibrido,
        }
        if self.init_strategy not in dispatch:
            raise ValueError(f"init_strategy inválida: '{self.init_strategy}'")
        if self.init_strategy == 'aleatorio':
            dispatch[self.init_strategy](n_features, rng)
        else:
            dispatch[self.init_strategy](rng)

    # ── Entrenamiento / predicción ─────────────────────────────────────────

    def _H(self, X):
        return _activacion(X @ self.W.T + self.b, self.act)

    def fit(self, X, y):
        rng            = np.random.default_rng(self.seed)
        self.n_classes = int(y.max()) + 1
        self._init_pesos(X.shape[1], rng)

        # ── Modo híbrido GPU float32 / CPU float64 ─────────────────────────
        # PASO 1: H (N×L) en GPU float32  → el paso caro, se beneficia de la GPU
        # PASO 2: H^T·H (L×L) y H^T·Y (L×16) se pasan a CPU float64 → solo MB
        # PASO 3: solve lineal en CPU float64 → máxima precisión numérica
        # Esto evita la paginación VRAM←→RAM que ocurría con float64 en GPU
        # y garantiza la misma precisión que el modo CPU puro.
        if _TORCH_DISPONIBLE and torch.cuda.is_available():
            try:
                device = torch.device('cuda')

                # — PASO 1: calcular H en GPU (float32, in-place) —
                X_t = torch.from_numpy(X).to(device).float()
                W_t = torch.from_numpy(self.W).to(device).float()
                b_t = torch.from_numpy(self.b).to(device).float()

                H_t = torch.addmm(b_t, X_t, W_t.t())   # N×L float32 en VRAM
                if self.act == 'relu':
                    H_t.clamp_(min=0)
                elif self.act == 'sigmoid':
                    H_t.clamp_(min=-500, max=500).sigmoid_()
                elif self.act == 'tanh':
                    H_t.tanh_()
                else:
                    raise ValueError(f"activation desconocida: '{self.act}'")

                # — PASO 2: productos reducidos en GPU → CPU float64 —
                y_t = torch.from_numpy(y.astype(np.int64)).to(device)
                Y_t = torch.zeros((len(y), self.n_classes),
                                  dtype=torch.float32, device=device)
                Y_t.scatter_(1, y_t.unsqueeze(1), 1.0)

                # H^T·H: (L×N)·(N×L) = L×L  → mueve solo L×L a CPU
                HtH = (H_t.t() @ H_t).cpu().numpy().astype(np.float64)
                # H^T·Y: (L×N)·(N×16) = L×16 → mueve solo L×16 a CPU
                HtY = (H_t.t() @ Y_t).cpu().numpy().astype(np.float64)

                # Sincronizar antes de mover datos a CPU para que el timer
                # de fit() capture el tiempo de cómputo GPU real.
                torch.cuda.synchronize()

                # Liberar H de VRAM inmediatamente (ya no se necesita)
                del H_t, Y_t, X_t, W_t, b_t, y_t
                torch.cuda.empty_cache()

                # — PASO 3: solve en CPU con float64 (precisión total) —
                A = HtH + np.eye(self.L, dtype=np.float64) / self.C
                self.beta = np.linalg.solve(A, HtY)   # float64

                # Guardar beta también en GPU float32 para predicción rápida
                self.beta_t = torch.from_numpy(
                    self.beta.astype(np.float32)).to(device)
                self._usar_gpu = True
                return

            except Exception as e:
                print(f"  [ELM Híbrido GPU] Error durante fit: {e}. "
                      f"Cayendo en fallback NumPy...")

        # ── Fallback NumPy (CPU, float64) ──────────────────────────────────
        self._usar_gpu = False
        H         = self._H(X)
        Y         = _one_hot(y, self.n_classes)
        A         = H.T @ H + np.eye(self.L) / self.C
        self.beta = np.linalg.solve(A, H.T @ Y)

    def predict(self, X):
        if _TORCH_DISPONIBLE and torch.cuda.is_available() and \
                getattr(self, '_usar_gpu', False):
            try:
                device = torch.device('cuda')
                X_t = torch.from_numpy(X).to(device).float()
                W_t = torch.from_numpy(self.W).to(device).float()
                b_t = torch.from_numpy(self.b).to(device).float()

                H_t = torch.addmm(b_t, X_t, W_t.t())
                if self.act == 'relu':
                    H_t.clamp_(min=0)
                elif self.act == 'sigmoid':
                    H_t.clamp_(min=-500, max=500).sigmoid_()
                elif self.act == 'tanh':
                    H_t.tanh_()
                else:
                    raise ValueError(f"activation desconocida: '{self.act}'")

                # beta_t en GPU float32 → predicción rápida
                preds = torch.argmax(H_t @ self.beta_t, dim=1)
                torch.cuda.synchronize()
                return preds.cpu().numpy().astype(np.int32)

            except Exception as e:
                print(f"  [ELM Híbrido GPU] Error durante predict: {e}. "
                      f"Cayendo en fallback NumPy...")

        return np.argmax(self._H(X) @ self.beta, axis=1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Clasificador para el benchmark
# ─────────────────────────────────────────────────────────────────────────────

class ClasificadorELM(ClasificadorBase):

    ESTRATEGIAS_VALIDAS = ('aleatorio', 'constelacion', 'voronoi', 'hibrido')

    def __init__(
        self,
        hidden_units        = 1000,
        C                   = 10.0,
        activation_function = 'relu',
        random_type         = 'normal',
        init_strategy       = 'aleatorio',
        sigma_init          = 0.5,
        hidden_units_grid   = None,
        C_grid              = None,
        activation_grid     = None,
        cv_folds            = 3,
        optimizar           = True,
        guardar_modelo      = False,
        dir_modelos         = "modelos/",
        seed                = 42,
        usar_cache          = True,
        dir_cache           = "hiperparametros/",
    ):
        super().__init__()
        if init_strategy not in self.ESTRATEGIAS_VALIDAS:
            raise ValueError(
                f"init_strategy='{init_strategy}' inválida. "
                f"Opciones: {self.ESTRATEGIAS_VALIDAS}")

        self.hidden_units        = hidden_units
        self.C                   = C
        self.activation_function = activation_function
        self.random_type         = random_type
        self.init_strategy       = init_strategy
        self.sigma_init          = sigma_init
        self.hidden_units_grid   = hidden_units_grid or [500, 1000, 2000, 5000, 10000]
        self.C_grid              = C_grid            or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.activation_grid     = activation_grid   or ['relu', 'tanh']
        self.cv_folds            = cv_folds
        self.optimizar           = optimizar
        self.guardar_modelo      = guardar_modelo
        self.dir_modelos         = dir_modelos
        self.seed                = seed
        self.usar_cache          = usar_cache
        self.dir_cache           = dir_cache
        self._modelo             = None
        self._mejores_params     = {}

    @property
    def nombre(self):
        sufijo = '' if self.init_strategy == 'aleatorio' \
                    else f'-{self.init_strategy[:3].capitalize()}'
        return f"ELM{sufijo}"

    @property
    def nombre_base(self):
        return "ELM"

    def _fit_interno(self, X_train, y_train):
        self._modelo = _ELMCore(
            hidden_units        = self.hidden_units,
            activation_function = self.activation_function,
            C                   = self.C,
            random_type         = self.random_type,
            seed                = self.seed,
            init_strategy       = self.init_strategy,
            sigma_init          = self.sigma_init,
        )
        self._modelo.fit(np.asarray(X_train, dtype=np.float64),
                         np.asarray(y_train, dtype=np.int32))
        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X):
        return self._modelo.predict(np.asarray(X, dtype=np.float64))

    def optimizar_hiperparametros(self, X_train, y_train):
        nombre_cache = f"elm_{self.init_strategy}_hiperparams.json"

        if self.usar_cache and self.dir_cache:
            ruta = os.path.join(self.dir_cache, nombre_cache)
            if os.path.exists(ruta):
                try:
                    with open(ruta, 'r', encoding='utf-8') as f:
                        datos = json.load(f)
                    params  = datos.get("params", {})
                    acc     = datos.get("cv_accuracy")
                    fecha   = datos.get("fecha", "?")
                    print(f"  [Cache] ELM-{self.init_strategy}: {params} "
                          f"CV={acc:.4f if acc else 'N/A'} ({fecha})")
                    self._aplicar_params(params)
                    self._mejores_params = params
                    return params
                except Exception as e:
                    print(f"  [Cache] Error ({e}), ejecutando GridSearch...")

        if not self.optimizar:
            return {}

        print(f"  [ELM-{self.init_strategy}] GridSearch {self.cv_folds}-fold "
              f"L={self.hidden_units_grid} C={self.C_grid} act={self.activation_grid}")

        N_cv = min(len(X_train), 50_000)
        rng  = np.random.default_rng(self.seed)
        idx  = rng.choice(len(X_train), N_cv, replace=False)
        X_cv = np.asarray(X_train[idx], dtype=np.float64)
        y_cv = np.asarray(y_train[idx], dtype=np.int32)

        skf      = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                   random_state=self.seed)
        mejor_acc    = -1.0
        mejor_params = {}
        total        = (len(self.hidden_units_grid) *
                        len(self.C_grid) *
                        len(self.activation_grid))
        contador = 0

        for act in self.activation_grid:
            for L in self.hidden_units_grid:
                for C in self.C_grid:
                    accs = []
                    for ft, fv in skf.split(X_cv, y_cv):
                        elm = _ELMCore(L, act, C,
                                       self.random_type,
                                       seed          = self.seed,
                                       init_strategy = self.init_strategy,
                                       sigma_init    = self.sigma_init)
                        elm.fit(X_cv[ft], y_cv[ft])
                        accs.append(np.mean(elm.predict(X_cv[fv]) == y_cv[fv]))
                    acc_mean  = float(np.mean(accs))
                    contador += 1
                    print(f"    [{contador:02d}/{total}] {act} L={L} C={C} "
                          f"acc={acc_mean:.4f}")
                    if acc_mean > mejor_acc:
                        mejor_acc    = acc_mean
                        mejor_params = {'hidden_units': L, 'C': C,
                                        'activation': act}

        self._aplicar_params(mejor_params)
        self._mejores_params = mejor_params
        print(f"  [ELM-{self.init_strategy}] Mejor: {mejor_params} "
              f"CV={mejor_acc:.4f}")

        if self.dir_cache:
            self._guardar_cache(mejor_params, mejor_acc, nombre_cache)
        return mejor_params

    def _aplicar_params(self, params):
        if 'hidden_units' in params: self.hidden_units = params['hidden_units']
        if 'C'            in params: self.C = params['C']
        if 'activation'   in params: self.activation_function = params['activation']

    def _calcular_flops(self) -> int:
        d = 2
        H = self._modelo.L
        # Capa de entrada: x @ W.T + b  →  H × (2d + 1) = H × 5 FLOPs (d=2)
        flops_H = H * (2 * d + 1)
        # Funcion de activacion: relu=1, tanh≈5, sigmoid≈10 FLOPs por neurona
        costo_act = {'relu': 1, 'tanh': 5, 'sigmoid': 10}
        flops_act = H * costo_act.get(self._modelo.act, 1)
        # Capa de salida: H @ beta, beta shape (H, 16)  →  16 × (2H) FLOPs
        flops_out = 2 * H * 16
        return flops_H + flops_act + flops_out + 15  # +15 argmax

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos,
                            f"modelo_ELM_{self.init_strategy}.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({'modelo': self._modelo,
                         'params': self._mejores_params}, f)

    def _guardar_cache(self, params, cv_accuracy, nombre_cache):
        os.makedirs(self.dir_cache, exist_ok=True)
        ruta = os.path.join(self.dir_cache, nombre_cache)
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({"params": params, "cv_accuracy": float(cv_accuracy),
                       "init_strategy": self.init_strategy,
                       "sigma_init": self.sigma_init,
                       "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=2)
        print(f"  [Cache] guardado: {ruta}")
