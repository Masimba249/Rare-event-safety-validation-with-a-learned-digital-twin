"""Small shared helpers: seeding, IO, timing, logging and a k-means."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

LOG = logging.getLogger("rsv")

_STREAMS = {
    "data": 1,
    "twin": 2,
    "twin_refit": 3,
    "mcts": 4,
    "cem": 5,
    "mc": 6,
    "is": 7,
    "sim2real": 8,
    "plant_reference": 9,
    "plots": 10,
    "misc": 11,
}


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def rng_for(master_seed: int, stream: str, index: int = 0) -> np.random.Generator:
    """Deterministic, independent generator for a named pipeline stage.

    Using :class:`numpy.random.SeedSequence` means adding a stage never shifts
    the random numbers used by the others, so results stay reproducible while
    the pipeline evolves.
    """
    if stream not in _STREAMS:
        raise KeyError("unknown rng stream '%s'" % stream)
    seq = np.random.SeedSequence([int(master_seed), _STREAMS[stream], int(index)])
    return np.random.default_rng(seq)


@contextmanager
def timed(label: str) -> Iterator[Dict[str, float]]:
    """Context manager logging wall-clock time; yields a dict with ``seconds``."""
    info: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - t0
        LOG.info("%s finished in %.1f s", label, info["seconds"])


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError("not JSON serialisable: %r" % type(obj))


def save_npz(path: str, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as fh:
        return {k: fh[k] for k in fh.files}


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #


def kmeans(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    iters: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plain k-means++ clustering (no sklearn dependency).

    Returns ``(centres, labels)``.  ``k`` is clipped to ``len(x)``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[0]
    k = int(max(1, min(k, n)))

    # --- k-means++ seeding --------------------------------------------- #
    centres = np.empty((k, x.shape[1]), dtype=float)
    centres[0] = x[rng.integers(n)]
    closest = np.sum((x - centres[0]) ** 2, axis=1)
    for j in range(1, k):
        total = float(closest.sum())
        if not np.isfinite(total) or total <= 0:
            centres[j] = x[rng.integers(n)]
        else:
            centres[j] = x[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, np.sum((x - centres[j]) ** 2, axis=1))

    labels = np.zeros(n, dtype=int)
    for _ in range(int(iters)):
        d2 = np.sum((x[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)
            else:  # re-seed an empty cluster on the worst-fit point
                far = int(np.argmax(np.min(d2, axis=1)))
                centres[j] = x[far]
    return centres, labels


def weighted_quantile(
    values: np.ndarray, quantiles: np.ndarray, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Weighted quantiles of ``values`` (used for reliability diagrams)."""
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if weights is None:
        return np.quantile(values, quantiles)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(quantiles, cw, v)


def batched(total: int, batch: int) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` index pairs covering ``range(total)``."""
    batch = int(max(1, batch))
    for start in range(0, int(total), batch):
        yield start, min(start + batch, int(total))


def fmt_prob(p: float) -> str:
    """Compact rendering of a (possibly very small) probability."""
    if p == 0:
        return "0"
    if p >= 1e-3:
        return "%.4g" % p
    return "%.3e" % p
