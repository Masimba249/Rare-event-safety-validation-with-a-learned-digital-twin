#!/usr/bin/env python3
"""ROS 2 node that executes one stress-test scenario on the real robot.

Takes a latent vector saved by the pipeline (``results/replay_round1.npz``),
decodes it with :mod:`rsv.scenario`, and runs the episode: the same waypoint
controller and latching obstacle-stop monitor as the simulator, with the
scenario's disturbances injected into the range readings and the command path.

What can and cannot be injected
-------------------------------
Injectable, because they live in software or in the fixture:

* range-finder noise, calibration bias and stale readings;
* actuation latency, as a delay line on the outgoing command;
* obstacle placement, which the node prints and waits for you to set.

Not injectable: the dynamics-residual channels.  A real floor cannot be asked to
slip by a prescribed amount on a prescribed control step.  The paired replay
therefore reproduces everything *except* the process noise, and
``rsv.sim2real`` reports a separate validation rate over repeated runs to
account for that.

Safety
------
These scenarios are selected to cause a collision.  Use a padded obstacle, keep
a hardware e-stop in hand, and keep the cruise speed low.  The node refuses to
run above the configured nominal speed.

    ros2 run rsv_hw scenario --ros-args \
        -p scenario_file:=results/replay_round1.npz -p index:=0 -p config:=configs/default.yaml

Reference implementation: not exercised by the test suite.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Range

from rsv.config import Config
from rsv.controller import SafetyController, wrap_angle
from rsv.scenario import ScenarioSpace


class ScenarioRunner(Node):
    """Runs one decoded scenario end to end and reports the outcome."""

    def __init__(self) -> None:
        super().__init__("rsv_scenario")
        self.declare_parameter("scenario_file", "results/replay_round1.npz")
        self.declare_parameter("index", 0)
        self.declare_parameter("config", "configs/default.yaml")
        self.declare_parameter("dry_run", False)

        self.cfg = Config.load(str(self.get_parameter("config").value))
        self.dt = self.cfg.scenario.dt
        self.dry_run = bool(self.get_parameter("dry_run").value)

        with np.load(str(self.get_parameter("scenario_file").value)) as fh:
            z = fh["z"][int(self.get_parameter("index").value)]

        self.space = ScenarioSpace(self.cfg.scenario, n_models=self.cfg.twin.n_members)
        self.dist = self.space.decode(z[None, :])
        self.ctrl = SafetyController(self.cfg.controller, self.dt)

        self._announce()

        self.latency = int(self.dist.latency[0])
        self.ring = np.zeros((self.latency + 1, 2))
        self.latched = False
        self.held = self.cfg.controller.max_range
        self.v_cmd_prev = 0.0
        self.step = 0
        self.min_clearance = float("inf")

        self._pose: Optional[tuple] = None
        self._range: Optional[float] = None

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Range, "/range", self._on_range, sensor_qos)
        self.create_subscription(PoseStamped, "/aruco/pose", self._on_pose, sensor_qos)
        self.timer = self.create_timer(self.dt, self._tick)

    # ------------------------------------------------------------------ #
    def _announce(self) -> None:
        """Print the physical setup this scenario requires before it starts."""
        d = self.dist
        log = self.get_logger()
        log.info("--- scenario setup -------------------------------------")
        log.info("  place the obstacle at x = %.3f m, y = %+.3f m" % (d.obstacle_x[0], d.obstacle_y[0]))
        log.info("  floor traction required: mu = %.3f" % d.mu[0])
        log.info("  injected range bias: %+.4f m" % d.range_bias[0])
        log.info("  injected actuation latency: %d control steps" % d.latency[0])
        log.info("  stale range readings scheduled: %d of %d steps"
                 % (int(d.dropout[0].sum()), d.dropout.shape[1]))
        log.info("  NOTE: process-noise channels cannot be injected on hardware")
        log.info("  padded obstacle and e-stop required: this scenario is expected to collide")
        log.info("--------------------------------------------------------")

        if self.cfg.controller.v_nominal > 1.0:
            raise SystemExit("refusing to run a collision scenario above 1 m/s")

    def _on_range(self, msg: Range) -> None:
        self._range = float(msg.range)

    def _on_pose(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        t = self.step
        if t >= self.cfg.scenario.horizon or self._pose is None or self._range is None:
            if t >= self.cfg.scenario.horizon:
                self._finish()
            return

        d = self.dist
        x, y, theta = self._pose

        # ---- perception, with the scenario's disturbances injected ------- #
        raw = float(
            np.clip(
                self._range + d.range_bias[0] + d.range_noise[0, t],
                0.0,
                self.cfg.controller.max_range,
            )
        )
        measured = self.held if bool(d.dropout[0, t]) else raw
        self.held = measured
        self.min_clearance = min(self.min_clearance, self._range)

        # ---- the safety function under test ------------------------------ #
        self.latched = bool(self.ctrl.monitor(np.array([measured]), np.array([self.latched]))[0])
        v_cmd = float(self.ctrl.speed_command(np.array([self.v_cmd_prev]), np.array([self.latched]))[0])
        w_cmd = float(self.ctrl.yaw_command(np.array([x]), np.array([y]), np.array([theta]),
                                            self.cfg.scenario.goal_x)[0])
        self.v_cmd_prev = v_cmd

        # ---- actuation latency ------------------------------------------- #
        slot = t % (self.latency + 1)
        self.ring[slot] = (v_cmd, w_cmd)
        applied = self.ring[(t - self.latency) % (self.latency + 1)]
        ready = t >= self.latency

        msg = Twist()
        msg.linear.x = float(applied[0]) if ready else 0.0
        msg.angular.z = float(applied[1]) if ready else 0.0
        if not self.dry_run:
            self.pub.publish(msg)
        self.step += 1

    def _finish(self) -> None:
        self.pub.publish(Twist())  # stop
        hit = self.cfg.scenario.robot_radius + self.cfg.scenario.obstacle_radius
        margin = self.min_clearance - hit
        self.get_logger().info(
            "scenario finished: minimum clearance %.4f m, margin %+.4f m -> %s"
            % (self.min_clearance, margin, "COLLISION" if margin < 0 else "no contact")
        )
        rclpy.shutdown()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = ScenarioRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.pub.publish(Twist())
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
