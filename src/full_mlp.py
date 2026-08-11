"""
A 2-layer machine learning pipeline with forward and backward passes in NumPy.

Architecture:  X (N,d) -> Linear -> tanh -> Linear -> yhat (N,1)
Loss:          mean squared error, L = (1/N) * sum((yhat - y)^2)
"""
from __future__ import annotations

import numpy as np


class TwoLayerMLP:
    def __init__(self, n_features: int, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)

        # Xavier/Glorot initialisation
        self.W1 = rng.normal(0, np.sqrt(1.0 / n_features), (n_features, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1))
        self.b2 = np.zeros(1)

        self.cache: dict = {}

    # Forward Pass 
    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, d) -> yhat: (N, 1). Caches intermediates for backward()."""
        Z1 = X @ self.W1 + self.b1        # (N, h)
        A1 = np.tanh(Z1)                  # (N, h)
        Z2 = A1 @ self.W2 + self.b2       # (N, 1)


        self.cache = {"X": X, "A1": A1}
        return Z2                          # identity output

    @staticmethod
    def loss(yhat: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error. Matches torch.nn.MSELoss(reduction='mean')."""
        return float(np.mean((yhat - y) ** 2))

    # Backward Pass
    def backward(self, yhat: np.ndarray, y: np.ndarray) -> dict:
        """Return dL/dW1, dL/db1, dL/dW2, dL/db2."""
        X, A1 = self.cache["X"], self.cache["A1"]
        N = X.shape[0]

        # Step 1: dL/dYhat for L = (1/N) * sum((yhat - y)^2)
        dZ2 = (2.0 / N) * (yhat - y)               # (N, 1)

        # Step 2: through Z2 = A1 @ W2 + b2
        dW2 = A1.T @ dZ2                            # (h, 1)
        db2 = dZ2.sum(axis=0)                       # (1,)
        dA1 = dZ2 @ self.W2.T                       # (N, h)

        # Step 3: through A1 = tanh(Z1);  tanh'(z) = 1 - tanh(z)^2
        dZ1 = dA1 * (1.0 - A1 ** 2)                 # (N, h)

        # Step 4: through Z1 = X @ W1 + b1
        dW1 = X.T @ dZ1                             # (d, h)
        db1 = dZ1.sum(axis=0)                       # (h,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    
    # Parameter access 
    def get_params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def step(self, grads: dict, lr: float) -> None:
        """Plain gradient descent: move every parameter downhill."""
        for name, param in self.get_params().items():
            param -= lr * grads[name]


    # Training loop
    def fit(self, X, y, X_val=None, y_val=None, epochs=200,
            lr=0.05, batch_size=256, seed=0, verbose=True):
        """Mini-batch gradient descent. Returns per-epoch loss history."""
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = {"train": [], "val": []}

        for epoch in range(epochs):
            # Shuffles rows within training set each epoch, 
            # so the mini-batches are different each time.
            idx = rng.permutation(n)

            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                Xb, yb = X[batch], y[batch]

                yhat = self.forward(Xb)
                grads = self.backward(yhat, yb)
                self.step(grads, lr)

            train_loss = self.loss(self.forward(X), y)
            history["train"].append(train_loss)

            if X_val is not None:
                history["val"].append(self.loss(self.forward(X_val), y_val))

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                msg = f"epoch {epoch:>4}  train {train_loss:.5f}"
                if X_val is not None:
                    msg += f"  val {history['val'][-1]:.5f}"
                print(msg)

        return history