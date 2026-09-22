"""Common interface for anything that can advance the robot's velocity state.

Both the ground-truth plant (:mod:`rsv.plant`) and the learned twin
(:mod:`rsv.twin`) expose the same one-step signature, so
:func:`rsv.rollout.simulate` is agnostic about which one it is driving.  That is
what makes a paired sim-versus-hardware comparison a one-line change.

The learned part of the model is deliberately the *hard* part -- how a velocity
command becomes actual ground motion through motor lag, deadband and tyre
traction.  Pose integration from ``(v, omega)`` is exact kinematics and is done
in the rollout, not learned.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Features fed to the learned dynamics model.  The two "error" features are
# redundant in principle but make the lag structure much easier to fit.
FEATURE_NAMES = (
    "v",
    "omega",
    "v_cmd",
    "omega_cmd",
    "mu",
    "v_err",
    "omega_err",
)
TARGET_NAMES = ("dv", "domega")

N_FEATURES = len(FEATURE_NAMES)
N_TARGETS = len(TARGET_NAMES)


def build_features(
    v: np.ndarray,
    omega: np.ndarray,
    v_cmd: np.ndarray,
    omega_cmd: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Assemble the ``(B, N_FEATURES)`` design matrix for the dynamics model."""
    v, omega, v_cmd, omega_cmd, mu = (
        np.atleast_1d(np.asarray(a, dtype=float))
        for a in (v, omega, v_cmd, omega_cmd, mu)
    )
    return np.stack(
        (v, omega, v_cmd, omega_cmd, mu, v_cmd - v, omega_cmd - omega), axis=-1
    )


class Dynamics:
    """One-step velocity dynamics.

    Subclasses implement :meth:`step`, mapping the current velocity state and a
    command to the velocity increment over one control period.
    """

    name = "dynamics"
    #: True when the model also reports a predictive standard deviation.
    probabilistic = False

    def step(
        self,
        v: np.ndarray,
        omega: np.ndarray,
        v_cmd: np.ndarray,
        omega_cmd: np.ndarray,
        mu: np.ndarray,
        eps: np.ndarray,
        model_index: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(dv, domega)`` for one control step.

        ``eps`` has shape ``(B, 2)`` and holds standardised residual draws: the
        same latent numbers drive the plant's process noise and the twin's
        predictive noise, which gives paired comparisons between them.

        ``model_index`` selects, per row, which ensemble member advances that
        episode.  The plant ignores it; the twin uses it so that a rollout
        marginalises over model uncertainty rather than fixing one member.
        """
        raise NotImplementedError

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return ``(mean, aleatoric_sd, epistemic_sd)`` for a feature batch.

        Only probabilistic models need to implement this; it is used for
        calibration diagnostics and the uncertainty maps, not by the rollout.
        """
        raise NotImplementedError
