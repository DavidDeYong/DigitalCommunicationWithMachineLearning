"""
clasificadores/redes_neuronales.py
===================================
Clasificadores basados en redes neuronales para demodulación 16-QAM.

Dos arquitecturas:
  - ClasificadorMLP        : red densa simple  (2 → 128 → 64 → 16)
  - ClasificadorRedProfunda: red densa profunda (2 → 256 → 128 → 64 → 32 → 16)

Implementadas con PyTorch para control total sobre la arquitectura,
función de pérdida y optimizador.
Si PyTorch no está disponible, se usa sklearn MLPClassifier como fallback.
"""

import numpy as np
import os
import time
from .base import ClasificadorBase


# ---------------------------------------------------------------------------
# Detección de backend disponible
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_DISPONIBLE = True
except ImportError:
    _TORCH_DISPONIBLE = False

try:
    from sklearn.neural_network import MLPClassifier as _SKLearnMLP
    from sklearn.preprocessing  import StandardScaler
    _SKLEARN_DISPONIBLE = True
except ImportError:
    _SKLEARN_DISPONIBLE = False


# ---------------------------------------------------------------------------
# Clase base interna para redes neuronales con PyTorch
# ---------------------------------------------------------------------------

class _RedNeuronalTorch(ClasificadorBase):
    """Base común para MLP y red profunda con PyTorch."""

    def __init__(
        self,
        capas_ocultas: tuple,
        epochs:        int,
        batch_size:    int,
        lr:            float,
        lr_patience:   int   = 10,
        lr_factor:     float = 0.5,
        lr_min:        float = 1e-5,
        guardar_modelo:bool  = False,
        dir_modelos:   str   = "modelos",
        seed:          int   = 42,
    ):
        super().__init__()
        self.capas_ocultas  = capas_ocultas
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.lr_patience    = lr_patience
        self.lr_factor      = lr_factor
        self.lr_min         = lr_min
        self.guardar_modelo = guardar_modelo
        self.dir_modelos    = dir_modelos
        self.seed           = seed

        self.modelo     = None
        self.scaler_    = None
        self._historial_loss = []

        if _TORCH_DISPONIBLE:
            torch.manual_seed(seed)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif _SKLEARN_DISPONIBLE:
            pass
        else:
            raise ImportError(
                "Se requiere PyTorch o scikit-learn para las redes neuronales. "
                "Instalar con: pip install torch  o  pip install scikit-learn"
            )

    def _construir_red_torch(self, n_entradas: int = 2, n_salidas: int = 16) -> "nn.Module":
        """Construye la red con las capas ocultas definidas."""
        capas = []
        prev  = n_entradas
        for n in self.capas_ocultas:
            capas += [nn.Linear(prev, n), nn.BatchNorm1d(n), nn.ReLU()]
            prev   = n
        capas.append(nn.Linear(prev, n_salidas))
        return nn.Sequential(*capas)

    def _fit_interno(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if _TORCH_DISPONIBLE:
            self._fit_torch(X_train, y_train)
        else:
            self._fit_sklearn(X_train, y_train)

    def _fit_torch(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler

        # Normalizar
        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X_train).astype(np.float32)

        X_t = torch.from_numpy(X_sc).to(self.device)
        y_t = torch.from_numpy(y_train.astype(np.int64)).to(self.device)

        dataset    = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.modelo   = self._construir_red_torch().to(self.device)
        criterio      = nn.CrossEntropyLoss()
        optimizador   = torch.optim.Adam(self.modelo.parameters(), lr=self.lr)
        scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizador,
                            patience=self.lr_patience,
                            factor=self.lr_factor,
                            min_lr=self.lr_min)

        self.modelo.train()
        self._historial_loss = []

        for epoch in range(self.epochs):
            loss_total = 0.0
            for X_batch, y_batch in dataloader:
                optimizador.zero_grad()
                logits = self.modelo(X_batch)
                loss   = criterio(logits, y_batch)
                loss.backward()
                optimizador.step()
                loss_total += loss.item() * len(X_batch)

            loss_media = loss_total / len(X_train)
            self._historial_loss.append(loss_media)
            scheduler.step(loss_media)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}  loss={loss_media:.4f}")

        if self.guardar_modelo:
            self._guardar_torch()

    def _fit_sklearn(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fallback con sklearn MLPClassifier."""
        from sklearn.preprocessing import StandardScaler
        print(f"  [{self.nombre}] Usando sklearn MLPClassifier (PyTorch no disponible)")

        self.scaler_ = StandardScaler()
        X_sc = self.scaler_.fit_transform(X_train)

        self.modelo = _SKLearnMLP(
            hidden_layer_sizes=self.capas_ocultas,
            activation='relu',
            solver='adam',
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            batch_size=self.batch_size,
            random_state=self.seed,
            verbose=False,
        )
        self.modelo.fit(X_sc, y_train)

    def _predict_interno(self, X: np.ndarray) -> np.ndarray:
        if _TORCH_DISPONIBLE and isinstance(self.modelo, nn.Module):
            return self._predict_torch(X)
        else:
            return self._predict_sklearn(X)

    def _predict_torch(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_.transform(X).astype(np.float32)
        X_t  = torch.from_numpy(X_sc).to(self.device)
        self.modelo.eval()
        with torch.no_grad():
            logits = self.modelo(X_t)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int32)

    def _predict_sklearn(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_.transform(X)
        return self.modelo.predict(X_sc).astype(np.int32)

    def _calcular_flops(self) -> int:
        d = 2
        capas = [d] + list(self.capas_ocultas) + [16]
        flops = 0
        for i in range(len(capas) - 1):
            in_f, out_f = capas[i], capas[i + 1]
            # Capa lineal: out_f × (2 × in_f) MACs + out_f bias
            flops += 2 * in_f * out_f + out_f
            if i < len(capas) - 2:
                # BatchNorm: 4 ops por unidad (resta media, div std, escala, sesgo)
                flops += out_f * 4
                # ReLU: 1 op por unidad
                flops += out_f
        # StandardScaler: 2 ops por feature
        flops += d * 2
        # argmax sobre 16 clases
        flops += 15
        return flops

    def _guardar_torch(self):
        import pickle
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos,
                            f"modelo_{self.nombre.replace(' ', '_')}.pt")
        torch.save(self.modelo.state_dict(), ruta)
        ruta_scaler = ruta.replace('.pt', '_scaler.pkl')
        with open(ruta_scaler, 'wb') as f:
            pickle.dump(self.scaler_, f)


# ---------------------------------------------------------------------------
# MLP Simple: 2 → 128 → 64 → 16
# ---------------------------------------------------------------------------

class ClasificadorMLP(_RedNeuronalTorch):

    def __init__(
        self,
        capas_ocultas:  tuple = (128, 64),
        epochs:         int   = 100,
        batch_size:     int   = 512,
        lr:             float = 1e-3,
        lr_patience:    int   = 10,
        lr_factor:      float = 0.5,
        lr_min:         float = 1e-5,
        guardar_modelo: bool  = False,
        dir_modelos:    str   = "modelos/",
        seed:           int   = 42,
    ):
        super().__init__(
            capas_ocultas=capas_ocultas, epochs=epochs,
            batch_size=batch_size, lr=lr,
            lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
            guardar_modelo=guardar_modelo, dir_modelos=dir_modelos, seed=seed
        )

    @property
    def nombre(self) -> str:
        return "MLP Simple"


# ---------------------------------------------------------------------------
# Red Profunda: 2 → 256 → 128 → 64 → 32 → 16
# ---------------------------------------------------------------------------

class ClasificadorRedProfunda(_RedNeuronalTorch):

    def __init__(
        self,
        capas_ocultas:  tuple = (256, 128, 64, 32),
        epochs:         int   = 200,
        batch_size:     int   = 512,
        lr:             float = 1e-3,
        lr_patience:    int   = 15,
        lr_factor:      float = 0.5,
        lr_min:         float = 1e-5,
        guardar_modelo: bool  = False,
        dir_modelos:    str   = "modelos/",
        seed:           int   = 42,
    ):
        super().__init__(
            capas_ocultas=capas_ocultas, epochs=epochs,
            batch_size=batch_size, lr=lr,
            lr_patience=lr_patience, lr_factor=lr_factor, lr_min=lr_min,
            guardar_modelo=guardar_modelo, dir_modelos=dir_modelos, seed=seed
        )

    @property
    def nombre(self) -> str:
        return "Red Profunda"
