"""Adaptive Stress Testing by Monte Carlo tree search.

Follows the formulation of Lee, Kochenderfer and colleagues: treat the search
for a failure as a sequential decision problem whose actions are the
*disturbances*, and whose reward rewards failures that are **likely**, not
merely failures.  The return of a path is

    R  =  sum_levels log p(a_level)   +   { 0                  if it collides
                                          { -lambda * rho      otherwise

where ``rho`` is the miss distance.  Because the disturbance space here is a
standard normal (:mod:`rsv.scenario`), ``sum_levels log p(a)`` is
``-||z||^2 / 2`` up to a constant, so "maximise the return subject to failing"
means exactly "find the failure of highest nominal likelihood".  That is the
number the importance-sampling stage needs: the dominant point of the failure
region.

Structure of the tree
---------------------
Level 0 chooses the episode's static disturbances (floor traction, obstacle
placement, range-finder bias, actuation latency, and which ensemble member
governs this episode).  Every deeper level commits ``mcts_depth_chunk`` control
steps of per-step disturbances.  Actions are continuous, so the tree grows by
progressive widening: a node may acquire a new child only once its visit count
justifies one, which stops the search from fanning out over a 566-dimensional
action space instead of going deep.

Two implementation notes that matter:

* Only the *decided* prefix is scored in the return.  Including the log-density
  of the random rollout actions, as a literal reading of the formulation would,
  injects a ``sqrt(2 * d_tail) / 2`` standard-deviation noise term -- about
  ``+-16`` nats here -- which would drown the few-nat differences the search is
  trying to resolve.
* Nodes keep a copy of the simulator state, so descending the tree resumes an
  episode rather than replaying it from time zero.  The bookkeeping that ties a
  tree level to a range of control steps is spelled out in ``_steps_after``: get
  it off by one and the actions the tree chooses apply to steps that have
  already been simulated, which fails silently -- the search still returns
  failures, they just are not the failures it thinks it found.  The test
  ``test_reported_failures_replay_exactly`` pins it down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..dynamics import Dynamics
from ..rollout import EpisodeSimulator, EpisodeState, dist_dict as _dist_dict
from ..scenario import ScenarioSpace


@dataclass
class Node:
    """One decision point in the AST tree."""

    level: int
    prefix_cost: float  # -log p of the decided actions, i.e. 0.5 * ||z_decided||^2
    state: Optional[EpisodeState]  # simulator state after the decided prefix
    z: np.ndarray  # latent vector with the decided blocks filled in
    visits: int = 0
    total: float = 0.0
    children: List["Node"] = field(default_factory=list)
    terminal: bool = False
    terminal_return: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.total / self.visits if self.visits else 0.0


@dataclass
class ASTResult:
    """Failures found by the search, ranked by nominal likelihood."""

    failures: np.ndarray  # (n, d) latent vectors that collided
    log_p: np.ndarray  # (n,) nominal log-density of each
    robustness: np.ndarray  # (n,) how deep the collision was
    best_z: Optional[np.ndarray]
    best_log_p: float
    iterations: int
    episodes: int
    tree_nodes: int
    history: List[Dict[str, float]] = field(default_factory=list)

    @property
    def n_failures(self) -> int:
        return int(self.failures.shape[0])


class AdaptiveStressTest:
    """MCTS over disturbance sequences, driven by the learned twin."""

    def __init__(
        self,
        dynamics: Dynamics,
        cfg: Config,
        space: ScenarioSpace,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.space = space
        self.rng = rng
        self.sim = EpisodeSimulator(cfg, dynamics)

        ac = cfg.ast
        self.chunk = int(ac.mcts_depth_chunk)
        self.n_levels = 1 + int(np.ceil(cfg.scenario.horizon / self.chunk))
        self.action_sd = float(ac.mcts_action_sd)
        self.static_action_sd = float(ac.mcts_static_action_sd)
        self.rollout_batch = int(max(1, ac.mcts_rollout_batch))
        self.c_uct = float(ac.mcts_c_uct)
        self.pw_k = float(ac.mcts_pw_k)
        self.pw_k_root = float(ac.mcts_pw_k_root)
        self.pw_alpha = float(ac.mcts_pw_alpha)
        self.miss_penalty = float(ac.mcts_miss_penalty)

        self._value_lo = np.inf
        self._value_hi = -np.inf
        self._episodes = 0
        self._nodes = 1

    # ------------------------------------------------------------------ #
    # Latent-block bookkeeping
    # ------------------------------------------------------------------ #
    def _block_slice(self, level: int) -> slice:
        """Latent indices decided at ``level``."""
        if level == 0:
            return slice(0, self.space.n_static)
        start = level - 1
        lo = self.space.n_static + start * self.chunk * self.space.n_per_step
        hi = min(
            lo + self.chunk * self.space.n_per_step,
            self.space.n_static + self.space.horizon * self.space.n_per_step,
        )
        return slice(lo, hi)

    def _steps_after(self, level: int) -> int:
        """Control step the episode reaches once block ``level`` has been applied.

        Block 0 is the static disturbances and advances nothing; block ``k >= 1``
        covers control steps ``[(k-1)*chunk, k*chunk)``, so applying it leaves the
        episode at step ``k*chunk``.  A node at level ``L`` has blocks ``0..L-1``
        applied and therefore sits at step ``(L-1)*chunk``.
        """
        return 0 if level == 0 else min(level * self.chunk, self.space.horizon)

    # ------------------------------------------------------------------ #
    # Tree operations
    # ------------------------------------------------------------------ #
    def _expand(self, node: Node) -> Node:
        """Add one child by sampling a fresh action block."""
        sl = self._block_slice(node.level)
        # The root chooses the episode's static disturbances, and those are
        # what decide whether a failure is reachable at all, so it explores
        # further into the tails than the per-step levels do.
        sd = self.static_action_sd if node.level == 0 else self.action_sd
        action = sd * self.rng.standard_normal(sl.stop - sl.start)

        z = node.z.copy()
        z[sl] = action
        cost = node.prefix_cost + 0.5 * float(action @ action)

        dist = self.space.decode(z[None, :])
        if node.level == 0:
            state = self.sim.init(dist)
        else:
            state = node.state.with_disturbances(dist)
        # Run exactly the steps this block governs -- not the next block's.
        self.sim.run(state, until=self._steps_after(node.level))

        gap = float(state.robustness[0])
        settled = bool(self.sim.finished(state, state.robustness)[0])
        is_last = node.level + 1 >= self.n_levels
        child = Node(
            level=node.level + 1,
            prefix_cost=cost,
            state=state,
            z=z,
            terminal=is_last or settled,
        )
        if child.terminal:
            child.terminal_return = self._terminal_return(cost, gap)
        node.children.append(child)
        self._nodes += 1
        return child

    def _terminal_return(self, prefix_cost: float, robustness: float) -> float:
        """AST return: likelihood of the decided actions, minus a miss penalty."""
        miss = 0.0 if robustness < 0.0 else self.miss_penalty * robustness
        return -prefix_cost - miss

    def _uct_select(self, node: Node) -> Node:
        """UCT over children, with returns rescaled to the range seen so far."""
        span = max(self._value_hi - self._value_lo, 1e-9)
        log_n = np.log(max(node.visits, 1))
        best, best_score = node.children[0], -np.inf
        for child in node.children:
            if child.visits == 0:
                return child
            q = (child.mean_value - self._value_lo) / span
            score = q + self.c_uct * np.sqrt(log_n / child.visits)
            if score > best_score:
                best, best_score = child, score
        return best

    def _widen(self, node: Node) -> bool:
        """Progressive widening rule, with a wider budget at the root.

        The root's children are candidate *static* disturbance settings, and a
        failure mode can occupy a very narrow band of them; too few candidates
        and the search never samples the band at all.  Deeper levels refine an
        episode that is already in play and do not need the same fan-out.
        """
        k = self.pw_k_root if node.level == 0 else self.pw_k
        budget = k * (max(node.visits, 1) ** self.pw_alpha)
        return len(node.children) < max(1, int(np.ceil(budget)))

    # ------------------------------------------------------------------ #
    def _rollout(
        self, node: Node
    ) -> Tuple[float, List[np.ndarray], List[float]]:
        """Finish the episode from ``node`` with nominal disturbances.

        Evaluates a batch of independent futures in one vectorised pass and
        averages their AST returns.  Averaging matters as much as the speed: the
        return of a single rollout is a very noisy estimate of a node's value
        when the outcome hinges on a few rare draws, and a noisy value makes the
        tree policy wander.
        """
        if node.terminal:
            rob = float(node.state.robustness[0])
            z = [node.z.copy()] if rob < 0.0 else []
            return node.terminal_return, z, [rob] if rob < 0.0 else []

        w = self.rollout_batch
        # Blocks 0..level-1 are decided, so the undecided tail begins where this
        # node's own block begins.
        tail = slice(self._block_slice(node.level).start, self.space.dim)
        n_tail = tail.stop - tail.start

        z = np.repeat(node.z[None, :], w, axis=0)
        z[:, tail] = self.rng.standard_normal((w, n_tail))

        state = node.state.repeat(w)
        state.dist = _dist_dict(self.space.decode(z))
        self.sim.run(state)
        self._episodes += w

        rob = state.robustness
        returns = np.where(
            rob < 0.0, -node.prefix_cost, -node.prefix_cost - self.miss_penalty * rob
        )
        hits = np.flatnonzero(rob < 0.0)
        return (
            float(returns.mean()),
            [z[i].copy() for i in hits],
            [float(rob[i]) for i in hits],
        )

    # ------------------------------------------------------------------ #
    def search(self, iterations: Optional[int] = None) -> ASTResult:
        """Run the tree search and return the failures it found."""
        n_iter = int(self.cfg.ast.mcts_iterations if iterations is None else iterations)
        root = Node(level=0, prefix_cost=0.0, state=None, z=np.zeros(self.space.dim))

        found_z: List[np.ndarray] = []
        found_rob: List[float] = []
        history: List[Dict[str, float]] = []
        best_cost = np.inf

        for it in range(n_iter):
            path = [root]
            node = root
            while not node.terminal:
                if self._widen(node) or not node.children:
                    node = self._expand(node)
                    path.append(node)
                    break
                node = self._uct_select(node)
                path.append(node)

            value, hit_z, hit_rob = self._rollout(node)

            for zz, rr in zip(hit_z, hit_rob):
                found_z.append(zz)
                found_rob.append(rr)
                best_cost = min(best_cost, 0.5 * float(zz @ zz))

            self._value_lo = min(self._value_lo, value)
            self._value_hi = max(self._value_hi, value)
            for n in path:
                n.visits += 1
                n.total += value

            if (it + 1) % max(1, n_iter // 40) == 0:
                history.append(
                    {
                        "iteration": it + 1,
                        "failures": len(found_z),
                        "best_log_p": float(-best_cost) if np.isfinite(best_cost) else float("nan"),
                        "tree_nodes": self._nodes,
                    }
                )

        return self._collect(found_z, found_rob, n_iter, history)

    # ------------------------------------------------------------------ #
    def _collect(
        self,
        found_z: List[np.ndarray],
        found_rob: List[float],
        iterations: int,
        history: List[Dict[str, float]],
    ) -> ASTResult:
        if not found_z:
            return ASTResult(
                failures=np.zeros((0, self.space.dim)),
                log_p=np.zeros(0),
                robustness=np.zeros(0),
                best_z=None,
                best_log_p=float("-inf"),
                iterations=iterations,
                episodes=self._episodes,
                tree_nodes=self._nodes,
                history=history,
            )
        z = np.asarray(found_z, dtype=float)
        rob = np.asarray(found_rob, dtype=float)
        log_p = self.space.log_p(z)
        order = np.argsort(-log_p)  # most likely failures first
        keep = order[: max(1, int(self.cfg.ast.mcts_top_k))]
        return ASTResult(
            failures=z[keep],
            log_p=log_p[keep],
            robustness=rob[keep],
            best_z=z[order[0]],
            best_log_p=float(log_p[order[0]]),
            iterations=iterations,
            episodes=self._episodes,
            tree_nodes=self._nodes,
            history=history,
        )


def run_ast(
    dynamics: Dynamics,
    cfg: Config,
    space: ScenarioSpace,
    rng: np.random.Generator,
    iterations: Optional[int] = None,
) -> ASTResult:
    """Convenience wrapper around :class:`AdaptiveStressTest`."""
    return AdaptiveStressTest(dynamics, cfg, space, rng).search(iterations=iterations)
