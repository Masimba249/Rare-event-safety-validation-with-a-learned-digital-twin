#!/usr/bin/env python3
"""ROS 2 node that records a driving mission in the schema :mod:`rsv.data` reads.

Subscribes to the command, wheel-odometry and overhead-camera topics, resamples
them onto the control clock, and writes ``mission_XXXX.csv``.  The output is
byte-for-byte the format produced by ``rsv.data.write_logs_csv``, so a real
campaign and the simulated one are interchangeable from the twin's point of
view.

Reference implementation: not exercised by the test suite, and topic names and
frames will need adjusting for your robot.

    ros2 run rsv_hw recorder --ros-args \
        -p mission:=7 -p surface_mu:=0.52 -p duration_s:=8.0 -p out_dir:=/data/logs
"""

from __future__ import annotations

import csv
import math
import os
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

COLUMNS = (
    "mission",
    "step",
    "time_s",
    "cmd_v",
    "cmd_w",
    "enc_v",
    "enc_w",
    "aruco_x",
    "aruco_y",
    "aruco_theta",
    "aruco_valid",
    "surface_mu",
)


def yaw_from_quaternion(q) -> float:
    """Planar heading from a quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class Recorder(Node):
    """Samples the three streams on a fixed clock and writes one CSV."""

    def __init__(self) -> None:
        super().__init__("rsv_recorder")
        self.declare_parameter("mission", 0)
        self.declare_parameter("surface_mu", 0.5)
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("duration_s", 8.0)
        self.declare_parameter("out_dir", "logs")
        self.declare_parameter("marker_timeout_s", 0.15)

        self.mission = int(self.get_parameter("mission").value)
        self.surface_mu = float(self.get_parameter("surface_mu").value)
        self.dt = float(self.get_parameter("dt").value)
        self.duration = float(self.get_parameter("duration_s").value)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.marker_timeout = float(self.get_parameter("marker_timeout_s").value)

        # Latest value from each stream; the control clock samples these.
        self._cmd = (0.0, 0.0)
        self._enc = (0.0, 0.0)
        self._pose: Optional[tuple] = None
        self._pose_stamp = 0.0

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, sensor_qos)
        self.create_subscription(PoseStamped, "/aruco/pose", self._on_aruco, sensor_qos)

        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, "mission_%04d.csv" % self.mission)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(COLUMNS)

        self.step = 0
        self.t0 = self._now()
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info("recording mission %d to %s" % (self.mission, self.path))

    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cmd(self, msg: Twist) -> None:
        self._cmd = (msg.linear.x, msg.angular.z)

    def _on_odom(self, msg: Odometry) -> None:
        self._enc = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _on_aruco(self, msg: PoseStamped) -> None:
        self._pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            yaw_from_quaternion(msg.pose.orientation),
        )
        self._pose_stamp = self._now()

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        """One control-period sample.

        A marker older than ``marker_timeout_s`` is recorded as invalid with its
        last value held, which is exactly how the offline estimator treats a
        dropped frame -- so the two agree about what a dropout means.
        """
        now = self._now()
        elapsed = now - self.t0
        if elapsed > self.duration:
            self.get_logger().info("mission %d complete (%d steps)" % (self.mission, self.step))
            self._fh.close()
            rclpy.shutdown()
            return

        fresh = self._pose is not None and (now - self._pose_stamp) <= self.marker_timeout
        x, y, theta = self._pose if self._pose is not None else (0.0, 0.0, 0.0)

        self._writer.writerow(
            [
                self.mission,
                self.step,
                "%.4f" % elapsed,
                "%.6f" % self._cmd[0],
                "%.6f" % self._cmd[1],
                "%.6f" % self._enc[0],
                "%.6f" % self._enc[1],
                "%.6f" % x,
                "%.6f" % y,
                "%.6f" % theta,
                int(bool(fresh)),
                "%.6f" % self.surface_mu,
            ]
        )
        self.step += 1


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._fh.closed:
            node._fh.close()
        node.destroy_node()


if __name__ == "__main__":
    main()
