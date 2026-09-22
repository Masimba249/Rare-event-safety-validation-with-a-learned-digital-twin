"""A small MLP with a heteroscedastic Gaussian head, in plain NumPy.

Each network outputs a mean and a log-variance per target and is trained by
minimising the Gaussian negative log-likelihood

    NLL = 0.5 * mean_i sum_k [ (y_ik - mu_ik)^2 / s_ik^2 + log s_ik^2 ]

which is what lets a single network report *where its own predictions are
noisy* (aleatoric uncertainty).  Disagreement between independently trained
networks supplies the complementary epistemic term (see
:mod:`rsv.models.ensemble`).

Plain NLL has a well-documented pathology: the ``1/sigma^2`` factor lets the
network dismiss any region it finds hard as "noisy" and stop fitting the mean
there.  That is precisely the wrong failure for this project, because the hard
region is the emergency stop.  So the loss is the beta-NLL of Seitzer et al.
(2022), which scales each term by a detached ``sigma^(2*beta)``: at ``beta = 1``
the gradient on the mean is exactly that of least squares, while the variance
head still learns.

Written directly rather than pulled from a deep-learning framework: the model is
tiny, the gradients are two pages long, and a dependency-free implementation
keeps the whole pipeline reproducible from ``pip install -r requirements.txt``.
The log-variance is bounded by the smooth softplus clamp from Chua et al.'s
PETS, which prevents the variance collapsing on easy samples early in training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


class MLP:
    """Fully connected network predicting ``(mean, log_var)`` per target."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden: Sequence[int],
        rng: np.random.Generator,
        activation: str = "tanh",
        min_sigma: float = 5e-3,
        max_sigma: float = 3.0,
        beta_nll: float = 0.5,
    ) -> None:
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.hidden = list(hidden)
        self.activation = activation
        self.beta_nll = float(beta_nll)
        self.min_logvar = 2.0 * float(np.log(min_sigma))
        self.max_logvar = 2.0 * float(np.log(max_sigma))

        sizes = [self.n_in] + self.hidden + [2 * self.n_out]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / (a + b))
            self.weights.append(rng.normal(0.0, scale, size=(a, b)))
            self.biases.append(np.zeros(b))
        # Start with a moderate, uniform predicted variance.
        self.biases[-1][self.n_out :] = -2.0
        self._inference_cache = None

    # ------------------------------------------------------------------ #
    def _act(self, a: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(a)
        if self.activation == "relu":
            return np.maximum(a, 0.0)
        raise ValueError("unknown activation %r" % self.activation)

    def _act_grad(self, h: np.ndarray) -> np.ndarray:
        """Derivative expressed in terms of the *post*-activation value."""
        if self.activation == "tanh":
            return 1.0 - h * h
        return (h > 0.0).astype(float)

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray, cache: Optional[List[np.ndarray]] = None):
        """Return ``(mean, logvar)``; append activations to ``cache`` if given."""
        h = np.atleast_2d(np.asarray(x, dtype=float))
        if cache is not None:
            cache.append(h)
        n_layers = len(self.weights)
        for i in range(n_layers - 1):
            h = self._act(h @ self.weights[i] + self.biases[i])
            if cache is not None:
                cache.append(h)
        out = h @ self.weights[-1] + self.biases[-1]
        mean = out[:, : self.n_out]
        logvar = self._clamp_logvar(out[:, self.n_out :])
        return mean, logvar

    def _clamp_logvar(self, raw: np.ndarray) -> np.ndarray:
        s1 = self.max_logvar - softplus(self.max_logvar - raw)
        return self.min_logvar + softplus(s1 - self.min_logvar)

    def _clamp_logvar_grad(self, raw: np.ndarray) -> np.ndarray:
        s1 = self.max_logvar - softplus(self.max_logvar - raw)
        return sigmoid(self.max_logvar - raw) * sigmoid(s1 - self.min_logvar)

    def _inference_weights(self):
        """Single-precision copies of the parameters, built once per update.

        Rollouts spend most of their time inside ``tanh``, and at single
        precision both it and the matrix products run about two and a half
        times faster -- which is the difference between a 200k-episode Monte
        Carlo taking two minutes and taking five.  Training stays in double
        precision; only the forward pass used by the simulator is narrowed,
        and the rollout state itself remains double.
        """
        if self._inference_cache is None:
            self._inference_cache = (
                [w.astype(np.float32) for w in self.weights],
                [b.astype(np.float32) for b in self.biases],
            )
        return self._inference_cache

    def invalidate_cache(self) -> None:
        """Drop the inference copies; call after any parameter change."""
        self._inference_cache = None

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, sigma)`` for a batch."""
        weights, biases = self._inference_weights()
        h = np.atleast_2d(np.asarray(x, dtype=np.float32))
        for i in range(len(weights) - 1):
            h = self._act(h @ weights[i] + biases[i])
        out = h @ weights[-1] + biases[-1]
        mean = out[:, : self.n_out].astype(np.float64)
        logvar = self._clamp_logvar(out[:, self.n_out :].astype(np.float64))
        return mean, np.exp(0.5 * logvar)

    # ------------------------------------------------------------------ #
    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Gaussian NLL and its gradients for one mini-batch."""
        cache: List[np.ndarray] = []
        h = np.atleast_2d(np.asarray(x, dtype=float))
        cache.append(h)
        n_layers = len(self.weights)
        for i in range(n_layers - 1):
            h = self._act(h @ self.weights[i] + self.biases[i])
            cache.append(h)
        raw = h @ self.weights[-1] + self.biases[-1]
        mean = raw[:, : self.n_out]
        raw_lv = raw[:, self.n_out :]
        logvar = self._clamp_logvar(raw_lv)

        inv_var = np.exp(-logvar)
        resid = mean - y
        # beta-NLL weighting: treated as a constant w.r.t. the parameters.
        beta_w = np.exp(self.beta_nll * logvar) if self.beta_nll else 1.0
        terms = 0.5 * (resid * resid * inv_var + logvar + LOG2PI)
        per_sample = np.sum(beta_w * terms, axis=1)

        if weights is None:
            w = np.full(len(per_sample), 1.0 / max(len(per_sample), 1))
        else:
            w = np.asarray(weights, dtype=float)
            w = w / max(w.sum(), 1e-12)
        loss = float(np.sum(w * per_sample))

        # ---- output-layer gradients ----------------------------------- #
        g_mean = w[:, None] * beta_w * resid * inv_var
        g_logvar = 0.5 * w[:, None] * beta_w * (1.0 - resid * resid * inv_var)
        g_raw_lv = g_logvar * self._clamp_logvar_grad(raw_lv)
        delta = np.concatenate((g_mean, g_raw_lv), axis=1)

        gw: List[np.ndarray] = [np.empty(0)] * n_layers
        gb: List[np.ndarray] = [np.empty(0)] * n_layers
        for i in range(n_layers - 1, -1, -1):
            h_in = cache[i]
            gw[i] = h_in.T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._act_grad(h_in)
        return loss, gw, gb

    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            state["W%d" % i] = w
            state["b%d" % i] = b
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        for i in range(len(self.weights)):
            self.weights[i] = np.asarray(state["W%d" % i], dtype=float)
            self.biases[i] = np.asarray(state["b%d" % i], dtype=float)
        self.invalidate_cache()


class Adam:
    """Adam with decoupled weight decay on the weight matrices only."""

    def __init__(self, model: MLP, lr: float, weight_decay: float = 0.0) -> None:
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.mw = [np.zeros_like(w) for w in model.weights]
        self.vw = [np.zeros_like(w) for w in model.weights]
        self.mb = [np.zeros_like(b) for b in model.biases]
        self.vb = [np.zeros_like(b) for b in model.biases]

    def step(
        self, gw: List[np.ndarray], gb: List[np.ndarray], lr_scale: float = 1.0
    ) -> None:
        self.model.invalidate_cache()
        self.t += 1
        lr = self.lr * lr_scale
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i in range(len(gw)):
            self.mw[i] = self.b1 * self.mw[i] + (1 - self.b1) * gw[i]
            self.vw[i] = self.b2 * self.vw[i] + (1 - self.b2) * gw[i] ** 2
            upd = (self.mw[i] / bc1) / (np.sqrt(self.vw[i] / bc2) + self.eps)
            self.model.weights[i] -= lr * (upd + self.weight_decay * self.model.weights[i])

            self.mb[i] = self.b1 * self.mb[i] + (1 - self.b1) * gb[i]
            self.vb[i] = self.b2 * self.vb[i] + (1 - self.b2) * gb[i] ** 2
            self.model.biases[i] -= lr * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + self.eps)


def train_mlp(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-3,
    weight_decay: float = 1e-5,
    sample_weight: Optional[np.ndarray] = None,
    verbose_every: int = 0,
) -> List[float]:
    """Train one network with a cosine learning-rate schedule.

    Returns the per-epoch training loss, which the twin report plots.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n = x.shape[0]
    opt = Adam(model, lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    batch_size = int(min(max(8, batch_size), n))

    for epoch in range(int(epochs)):
        order = rng.permutation(n)
        lr_scale = 0.5 * (1.0 + np.cos(np.pi * epoch / max(epochs - 1, 1)))
        lr_scale = 0.05 + 0.95 * lr_scale
        total = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            sel = order[start : start + batch_size]
            w = None if sample_weight is None else sample_weight[sel]
            loss, gw, gb = model.loss_and_grads(x[sel], y[sel], weights=w)
            opt.step(gw, gb, lr_scale=lr_scale)
            total += loss
            n_batches += 1
        history.append(total / max(n_batches, 1))
        if verbose_every and (epoch + 1) % verbose_every == 0:
            print("  epoch %4d  nll %.4f" % (epoch + 1, history[-1]))
    return history
