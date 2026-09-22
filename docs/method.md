# The statistical argument

This note sets out what the pipeline actually claims, what each claim rests on,
and where it can fail. The README covers the engineering; this covers the
reasoning.

## 1. What is being estimated

Let `z ∈ R^d` be the episode's disturbance vector with the operational law
`p(z) = N(0, I_d)`, and let `F(z) ∈ {0,1}` be the event that the robot's body
touches the obstacle during the episode. The target is

```
    p_fail = E_p[ F(z) ] = ∫ F(z) p(z) dz
```

Two things are worth stating plainly.

**`p` is a modelling choice, not a fact.** It encodes how often the floor is
slippery, how often a range reading goes stale, how far the obstacle placement
drifts. Every number this pipeline produces is conditional on it. Changing
`scenario.mu_sd` changes the answer, legitimately. The defensible claim is
"under this operational profile, the failure probability is …", never "the
failure probability is …".

**`F` is evaluated by a model.** In `stage_estimate` the episode runs on the
learned twin, so what is estimated is `p_fail` *under the twin*. Stage 5 is the
only thing that connects it to the robot.

## 2. Why the parameterisation is a standard normal

Every physical disturbance is a deterministic transform of `z`: affine for
Gaussian quantities, an inverse CDF on `Φ(z)` for discrete ones. Four
consequences, all load-bearing:

1. `log p(z) = -‖z‖²/2 - (d/2) log 2π` exactly. Importance weights are computed,
   not estimated — no density model sits between the estimator and the truth.
2. The adaptive stress testing objective `Σ_t log p(a_t)` becomes `-‖z‖²/2` plus
   a constant, so "most likely failure" is unambiguous and comparable across
   scenarios.
3. The cross-entropy method and Gaussian-mixture proposals live naturally in an
   unbounded isotropic space; no boundary handling, no reparameterisation.
4. A scenario replayed on hardware and in the twin shares one `z`, so the
   comparison is paired (common random numbers) and a disagreement is model
   error rather than sampling noise.

The cost is that the dynamics residual is *part of* `z`. The twin's predictive
noise is therefore integrated over as a component of the probability space,
which is correct only insofar as the twin's aleatoric estimate is calibrated —
hence the calibration reporting in `stage_fit`, and the measurement-noise
deconvolution described below.

## 3. Naive Monte Carlo, and why it is hopeless here

`p̂ = (1/N) Σ F(z_i)` with `z_i ~ p` is unbiased with variance `p(1-p)/N`.
Reaching relative error `ε` takes

```
    N ≈ (1 - p) / (p ε²)
```

At `p = 2 × 10⁻⁴` and `ε = 0.1`, that is about 500,000 episodes. At `p = 10⁻⁶`
it is 10⁸. The scaling is `1/p`, which is why direct testing does not work for
rare events and why every safety-critical field reaches for a variance-reduction
method instead.

Intervals at these rates need care. The Wald interval can put the lower bound
below zero and gives a zero-width interval when nothing fails. The pipeline
reports Clopper-Pearson, and when `k = 0` reports the only honest thing —
the one-sided bound `p < 1 - α^(1/N)`.

## 4. Importance sampling

Draw `z_i ~ q` and reweight:

```
    p̂_IS = (1/N) Σ w_i F(z_i),      w_i = p(z_i) / q(z_i)
```

unbiased for any `q` whose support covers `{F = 1}`. The variance is
`Var_q[w F] / N`, minimised at the zero-variance proposal
`q*(z) ∝ F(z) p(z) = p(z | failure)`. Everything in `rsv/estimate/` is an
attempt to approximate `q*` without ever being able to sample from it.

### 4.1 Bounded weights

The proposal is a *defensive* mixture (Hesterberg 1995):

```
    q(z) = α p(z) + (1-α) Σ_k β_k N(z; μ_k, Σ_k)
```

Since `q ≥ α p` pointwise, `w = p/q ≤ 1/α` everywhere. A bounded weight times a
bounded indicator has finite variance by construction, which is what licenses a
central-limit interval at all. Unbounded-weight importance sampling of a rare
event can have infinite variance and produce intervals that are confidently
wrong; that is the classic failure of the technique and it is not detectable
from the sample.

The defensive component is not free. It contributes roughly
`p_uncovered / (α N)` to the variance, where `p_uncovered` is the failure mass
the tilted components do not reach. Good coverage makes it cheap; poor coverage
makes it both expensive *and* invisible — see §4.4.

### 4.2 Dimension screening

Fitting a diagonal Gaussian to a set of failures estimates a mean and a standard
deviation in all `d = 566` coordinates, of which perhaps twenty carry signal.
The rest absorb sampling noise, and in high dimension those individually
negligible errors compound in the normalising constant. Shrinking `σ` to 0.45 in
every coordinate shifts `log q` by `566 log(1/0.45) ≈ 450` nats; `p/q`
underflows and the estimator silently degenerates to its defensive component,
which is *worse* than naive Monte Carlo at the same `N`.

`rsv/estimate/screening.py` soft-thresholds each coordinate against its own
standard error:

```
    shrink_j = max(0, 1 - c · se_j / |mean_j|)
```

A coordinate with real signal keeps almost all of its shift; one with only noise
returns to `N(0,1)` and contributes nothing to `log(p/q)`. The threshold is
looser during search than for the final proposal: leaking a few hundredths of a
standard deviation into irrelevant coordinates costs well under a nat, while
screening hard enough to zero them stalls a cross-entropy search before the real
shifts have grown.

### 4.3 Finding the modes, then covering them

**Clustering must happen in the right space.** In 566 dimensions every pair of
failures sits at nearly the same Euclidean distance — the irrelevant coordinates
contribute a common `√(2d)` — so k-means on raw latents returns arbitrary
groups. Clustering runs in the subspace scored by

```
    score_j = mean_j² + (var_j - 1)²
```

The variance term is essential: a mode such as "the obstacle was offset far
enough to fall outside the beam" is symmetric in the sign of the offset, so its
mean shift is zero and a mean-only score would discard the coordinate that
defines it.

**Search bias must be undone.** The failures a stress test returns are drawn
from wherever the search drifted, not from `p(z|failure)`. Reweighting them by
`p/q_search` is correct in principle but degenerate in practice — in one run
here, 4 of 17,836 failures carried all the weight, and the proposal fitted to
those 4 covered only one side of a symmetric mode. Two mitigations: the weights
are **tempered** (`w^β`, with `β` the largest exponent keeping the effective
sample size above a floor), and the search failures are used only to *aim* the
first proposal. Subsequent rounds draw from proposals of known density, pool
across rounds under the balance heuristic

```
    w_i = p(z_i) / Σ_r (n_r / N) q_r(z_i)
```

and refit on a monotonically growing sample. The reported estimate comes from a
final independent batch, so no adaptation data enters it.

**Coverage is constructed, not hoped for.** Searching for a mode is not the same
as covering it: a cross-entropy restart seeded toward one tail does not have to
stay there, and in this system the restart aimed at a positive lateral offset
drifts back to the traction mode because that failure is easier to reach. So the
proposal permanently carries one component per static channel per direction —
the systematic single-factor hypotheses an engineer would enumerate by hand.
They cost a fixed small share of the samples and remove an entire class of
silent failure.

### 4.4 The audit that matters

A poorly covering proposal does not produce a wide interval. It produces a
*narrow* interval around the wrong number, because the samples that would reveal
the gap — defensive draws landing in the uncovered mode — are too rare to occur.
Effective sample size, maximum weight share and agreement with a naive run all
look healthy. In this project an early version reported a value 29% low with a
95% interval that excluded the truth, and none of those diagnostics fired.

`coverage_diagnostic` audits it directly. The failures found by the naive Monte
Carlo run are an unbiased sample from the failure region under `p`, which is
exactly the sample needed. For each, the coverage ratio

```
    r(z) = q(z) / (α p(z))  ≥ 1
```

measures how much better the full proposal is at producing that failure than its
defensive component alone. `r ≈ 1` means that failure is reached only by luck.
The fraction of probe failures with `r < 2` estimates the share of failure mass
in that condition, with a Clopper-Pearson interval for the small counts a naive
run provides. A few percent is fine; tens of percent means the headline interval
is not to be believed and the stress testing has to go back for the missing mode.

## 5. From a statement about the model to a statement about the robot

Everything above estimates `p_fail` under the twin. Three measurements connect
it to hardware.

**Validation rate.** Predicted failures are replayed on the robot with the same
`z`. Since a real floor cannot be made to slip on command, the residual channels
cannot be injected; the paired replay holds everything else fixed, and repeats
with fresh noise give a per-scenario hardware failure *probability* rather than
one coin flip.

**Control group.** Nominal scenarios are replayed alongside. Without this, a
robot that collides with everything would score a perfect validation rate.

**The converse.** A twin can validate perfectly and still miss most of the ways
the robot actually fails. Real hardware failures are replayed in the twin,
several times each because the twin is stochastic, and the reproduction rate is
reported.

The replays are logged through the same measurement chain as the original
campaign and appended to the training set (DAgger). The whole estimate is
recomputed; the round-1 versus round-2 difference measures how much the hardware
loop was worth.

### 5.1 The aggregation is not free

Aggregating on-policy data has a cost that this experiment makes very visible.
The scenarios worth replaying are, by construction, the tail ones: low traction,
emergency stops. Append only those and every new row comes from a narrow slice
of the state space, so the refitted model buys accuracy there by losing it
elsewhere. Measured here, with the replayed rows at a 44% share of the training
set and almost all of them tail episodes:

* braking error *below* the originally logged traction range: 0.52 → 0.21 m/s²
  (the repair the loop is for);
* braking error *above* it: 0.37 → 0.44 m/s² (the cost);
* and with the replays replicated 3x instead of 1x — a 58% share — the cost
  overwhelms the repair: the refitted twin estimated a failure probability
  **27 times too high**.

Two things keep it in hand, both of which a real campaign should do anyway.
Replication is not used: DAgger aggregates datasets, it does not reweight them.
And the replay campaign is balanced — twice as many nominal scenarios as
predicted failures — so the added data describes ordinary driving as well as the
tail. That leaves a genuine residual trade: one round of the loop moves the
error in the tail down and the error elsewhere up a little, and the estimate
moves past the truth rather than onto it. Iterating the loop, rather than
running it once, is what would settle that; §6 lists it as a limitation rather
than pretending otherwise.

## 6. Where this can still be wrong

* **`p` is assumed.** Everything is conditional on the operational profile.
* **The twin's aleatoric noise is integrated over as truth.** If it is
  miscalibrated in the failure region — where, by construction, there is least
  data — the estimate inherits that error. The calibration report covers the
  held-out distribution, not the tail.
* **Coverage is audited against a finite probe.** With a few dozen naive
  failures, the audit has a wide interval of its own. It can catch a 30% gap; it
  cannot certify a 2% one.
* **Epistemic uncertainty is ensemble disagreement**, which is a heuristic. Five
  networks trained on the same data can agree confidently and all be wrong in
  the same direction.
* **Rounds are not iterated to convergence.** One hardware loop is run, and one
  round is not enough to settle the trade in §5.1: it corrects the direction of
  the error without landing on the truth. A real campaign would iterate until
  the round-to-round change was small and report that as its own convergence
  criterion.
