"""
clasificadores/sgd_nystroem.py
==============================
Clasificador SGD con aproximacion de kernel Nystroem para demodulacion 16-QAM.

Combina dos componentes de scikit-learn, ambos sin supuestos distribucionales:

  1. Nystroem (sklearn.kernel_approximation):
     Mapea los 2 features IQ a un espacio de alta dimension (n_components)
     que aproxima el kernel RBF. Costo O(N * n_components) vs O(N^2) del SVM RBF.

  2. SGDClassifier (sklearn.linear_model):
     Clasificador lineal en el espacio transformado, entrenado por gradiente
     estocastico. Costo O(N) por epoca. Converge en pocas pasadas.

Resultado: captura fronteras no-lineales equivalentes al SVM RBF pero con
entrenamiento lineal en N. Sin supuestos sobre la distribucion del ruido.

VENTAJA CLAVE PARA ADAPTABILIDAD:
  SGDClassifier soporta partial_fit() para entrenamiento online.
  reentrenar_incremental() actualiza los pesos con datos del nuevo canal
  sin resetear el modelo entrenado previamente.
  La transformacion Nystroem se mantiene fija (coherencia entre versiones).

  Caso de uso: canal pasa de AWGN a ruido con componente impulsiva.
  Con ~500 simbolos de trama piloto nueva, el modelo se adapta en milisegundos.

No requiere instalacion adicional: todo en scikit-learn.
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


class ClasificadorSGDNystroem(ClasificadorBase):

    def __init__(
        self,
        n_components    = 300,
        gamma           = 0.5,
        alpha           = 1e-4,
        max_iter        = 50,
        n_comp_grid     = None,
        gamma_grid      = None,
        cv_folds        = 3,
        optimizar       = True,
        guardar_modelo  = False,
        dir_modelos     = "modelos/",
        seed            = 42,
    ):
        """
        Parameters
        ----------
        n_components : dimension del espacio Nystroem.
                       Mayor -> mejor aproximacion del kernel, mas lento en inferencia.
                       Rango recomendado para 16-QAM (entrada 2D): 100-500.
        gamma        : parametro del kernel RBF (igual interpretacion que en SVM RBF).
        alpha        : regularizacion L2 del SGD. Controla overfitting.
        max_iter     : epocas de entrenamiento SGD.
        """
        super().__init__()
        self.n_components   = n_components
        self.gamma          = gamma
        self.alpha          = alpha
        self.max_iter       = max_iter
        self.n_comp_grid    = n_comp_grid or [100, 300, 500]
        self.gamma_grid     = gamma_grid  or [0.1, 0.5, 1.0, 2.0]
        self.cv_folds       = cv_folds
        self.optimizar      = optimizar
        self.guardar_modelo = guardar_modelo
        self.dir_modelos    = dir_modelos
        self.seed           = seed

        # Componentes separados (no Pipeline) para poder llamar partial_fit
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

        print(f"  [SGD-Nystroem] Optimizando hiperparametros "
              f"(GridSearch {self.cv_folds}-fold)...")

        N_cv = min(len(X_train), 30_000)
        idx  = np.random.choice(len(X_train), N_cv, replace=False)
        X_cv, y_cv = X_train[idx], y_train[idx]

        # Pipeline para que Nystroem se ajuste solo sobre el fold de train en cada split
        pipe = Pipeline([
            ('scaler',   StandardScaler()),
            ('nystroem', Nystroem(kernel='rbf', random_state=self.seed)),
            ('sgd',      SGDClassifier(
                loss='modified_huber', alpha=self.alpha,
                max_iter=self.max_iter, random_state=self.seed, n_jobs=-1,
            )),
        ])
        param_grid = {
            'nystroem__n_components': self.n_comp_grid,
            'nystroem__gamma':        self.gamma_grid,
        }
        gs = GridSearchCV(pipe, param_grid, cv=self.cv_folds,
                          scoring='accuracy', n_jobs=-1, verbose=0)
        gs.fit(X_cv, y_cv)

        self._mejores_params = {
            'n_components': gs.best_params_['nystroem__n_components'],
            'gamma':        gs.best_params_['nystroem__gamma'],
        }
        self.n_components = self._mejores_params['n_components']
        self.gamma        = self._mejores_params['gamma']
        print(f"  [SGD-Nystroem] Mejores params: {self._mejores_params}")
        return self._mejores_params

    def _fit_interno(self, X_train, y_train):
        # 1. Normalizar
        X_sc = self._scaler.fit_transform(X_train)

        # 2. Transformacion Nystroem (ajustar solo sobre datos de entrenamiento)
        self._nystroem = Nystroem(
            kernel       = 'rbf',
            gamma        = self.gamma,
            n_components = self.n_components,
            random_state = self.seed,
        )
        X_ny = self._nystroem.fit_transform(X_sc)

        # 3. Clasificador SGD con loss Huber modificada
        #    'modified_huber' es la unica opcion que: (a) es robusta a outliers,
        #    (b) soporta predict_proba, (c) soporta partial_fit
        self._sgd = SGDClassifier(
            loss         = 'modified_huber',
            alpha        = self.alpha,
            max_iter     = self.max_iter,
            tol          = 1e-4,
            random_state = self.seed,
            n_jobs       = -1,
        )
        self._sgd.fit(X_ny, y_train)

        if self.guardar_modelo:
            self._guardar()

    def _predict_interno(self, X):
        X_sc = self._scaler.transform(X)
        X_ny = self._nystroem.transform(X_sc)
        return self._sgd.predict(X_ny).astype(np.int32)

    def reentrenar_incremental(self, X_nuevo, y_nuevo, n_epochs=5):
        """
        Adapta el modelo online a nuevas condiciones del canal.

        Usa partial_fit sobre los datos del nuevo escenario.
        La transformacion Nystroem se mantiene fija (no se reajusta),
        lo que garantiza coherencia entre el modelo base y la actualizacion.
        Solo se actualizan los pesos del clasificador lineal SGD.

        Parameters
        ----------
        X_nuevo  : simbolos IQ de trama piloto del nuevo escenario de canal
        y_nuevo  : etiquetas verdaderas correspondientes
        n_epochs : pasadas sobre los datos nuevos (default: 5)

        Ejemplo de uso:
            # Entrenamiento inicial (canal AWGN)
            clf.fit(X_piloto_awgn, y_piloto)

            # El canal cambio (ruido impulsivo, fading, etc.)
            clf.reentrenar_incremental(X_piloto_nuevo, y_piloto_nuevo, n_epochs=5)

            # BER en el nuevo canal
            ber = calcular_ber(clf.predict(X_test_nuevo), y_test_nuevo)
        """
        if self._sgd is None:
            raise RuntimeError("[SGD-Nystroem] Llamar a fit() antes de reentrenar.")

        print(f"  [SGD-Nystroem] Adaptacion incremental: "
              f"{len(X_nuevo)} muestras, {n_epochs} epocas...")

        X_sc     = self._scaler.transform(X_nuevo)
        X_ny     = self._nystroem.transform(X_sc)
        clases   = np.arange(16)

        for epoch in range(n_epochs):
            idx = np.random.permutation(len(X_ny))
            self._sgd.partial_fit(X_ny[idx], y_nuevo[idx], classes=clases)

        print(f"  [SGD-Nystroem] Adaptacion completada.")

    def _guardar(self):
        os.makedirs(self.dir_modelos, exist_ok=True)
        ruta = os.path.join(self.dir_modelos, "modelo_SGD_Nystroem.pkl")
        with open(ruta, 'wb') as f:
            pickle.dump({
                'scaler':   self._scaler,
                'nystroem': self._nystroem,
                'sgd':      self._sgd,
                'params':   self._mejores_params,
            }, f)
