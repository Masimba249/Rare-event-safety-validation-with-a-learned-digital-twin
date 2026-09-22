# Running this on a real robot

Everything in `src/rsv/` is written against two interfaces and nothing else:

| Interface | Simulated by | On real hardware |
|---|---|---|
| Recorded driving data | `rsv.data.collect_logs` | `rsv_hw/recorder_node.py` |
| Executing a scenario | `rsv.rollout.simulate(..., Plant(...))` | `rsv_hw/scenario_node.py` |

`rsv.plant` is a stand-in for the robot: it produces the logs the twin learns
from, and it executes the replays in the sim-to-real stage. Swapping in a real
robot means replacing those two calls, not editing the estimators. Nothing in
the twin, the stress testers or the importance sampling knows which one it is
talking to.

## What the robot has to provide

A differential-drive base publishing wheel odometry, an overhead camera tracking
an ArUco marker on the robot's top plate, and a forward-facing range sensor.
The obstacle is a cylinder of known radius on a floor whose traction coefficient
you can measure (see "Measuring traction" below).

Concretely, at the control rate (20 Hz by default, `scenario.dt = 0.05`):

| Topic | Type | Used for |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | the command actually sent, logged as `cmd_v`, `cmd_w` |
| `/odom` | `nav_msgs/Odometry` | wheel-odometry velocities, logged as `enc_v`, `enc_w` |
| `/aruco/pose` | `geometry_msgs/PoseStamped` | overhead camera pose, logged as `aruco_x/y/theta` |
| `/scan` or `/range` | `sensor_msgs/LaserScan` or `Range` | the clearance the obstacle-stop monitor acts on |

## The data contract

Both the simulated and the real recorder write one CSV per mission, named
`mission_XXXX.csv`, with exactly these columns (`rsv.data.CSV_COLUMNS`):

```
mission, step, time_s, cmd_v, cmd_w, enc_v, enc_w,
aruco_x, aruco_y, aruco_theta, aruco_valid, surface_mu
```

* `time_s` — seconds from the start of the mission, one row per control step.
* `cmd_v`, `cmd_w` — the command *as sent*, in m/s and rad/s. Log what left the
  controller, not what the planner wanted: the twin is learning the response to
  the former.
* `enc_v`, `enc_w` — wheel odometry. These are recorded but **not** used as
  regression targets, because wheels over-read while spinning and under-read
  while skidding, and skidding is exactly the regime the safety case depends on.
  `rsv.data.estimator_comparison` quantifies that on your own logs.
* `aruco_x`, `aruco_y`, `aruco_theta` — overhead-camera pose in the room frame,
  metres and radians. This is what the twin is fitted to.
* `aruco_valid` — 0 when the marker was not detected in that frame; the
  estimator holds the previous value rather than interpolating.
* `surface_mu` — the traction coefficient of the floor for this mission.

Drop such files into any directory and `rsv.data.read_logs_csv` will load them;
from there `build_dataset` and `rsv.cli fit` work unchanged.

## Measuring traction

`surface_mu` is a per-mission covariate the twin uses as an input, so it has to
be measured, not guessed. The cheap standard method: from a known speed `v0`,
command an emergency stop on that floor and measure the stopping distance `d`
from the overhead camera. Then

    mu = v0^2 / (2 * g * d)

Average a few runs per surface. Two practical notes:

* The estimate is only valid if the stop is actually traction limited, which is
  what an instantaneous zero command achieves — a gentle ramp measures your
  controller, not your floor.
* Log the *measured* value per mission rather than a nominal per-surface value.
  Mis-recording this feature biases the twin exactly where it matters.

## Collecting the driving campaign

`_command_profiles` in `rsv.data` describes the campaign the simulated robot
drives, and a real campaign should mirror it:

* ordinary transport missions with speed set-points ramped at the controller's
  normal acceleration limit;
* **roughly 30% deliberate emergency-brake tests** — accelerate to a target
  speed, hold, command zero in a single step, repeat, sweeping the target speed
  across the operating range.

That last group is not optional. The obstacle stop is a traction-limited
manoeuvre, and a campaign of gentle driving identifies the traction limit only
weakly. The project's central finding is what happens at the edge of the
surfaces you did log, so log deliberately across as many floors as you can get:
the twin is trustworthy inside that range and confidently wrong outside it.

## Executing a scenario

`scenario_node.py` takes one latent vector, decodes it with `rsv.scenario`, and
runs the episode on the robot, injecting the disturbances in software:

* the obstacle's commanded placement (set by hand, or by a fixture, before the
  run — the node prints the target position and waits);
* range-finder noise, calibration bias and dropped readings, added to the
  measured range before the monitor sees it;
* actuation latency, as a delay line on the outgoing command.

The dynamics residual channels of the latent vector are **not** injectable: you
cannot ask a real floor to slip by a prescribed amount on a prescribed step.
That is why `rsv.sim2real.replay_on_hardware` reports two validation rates — one
paired replay, and the mean over several repeats with whatever noise the robot
actually produced.

## Safety

These scenarios are chosen to make the robot collide. Run them with a padded
obstacle, a clear floor, a hardware e-stop in someone's hand, and a speed low
enough that contact is harmless. The scenario node refuses to start above
`controller.v_nominal` for that reason.

The node files here are reference implementations: they are written against the
ROS 2 Python API, are not executed by the test suite, and will need your robot's
topic names and frames.
