#!/usr/bin/env python3
"""
PA4 Path Planner Node
=====================
State machine: TURN → SENSE → (REPLAN if obstacle) → DWA-MOVE
              repeat per waypoint until goal or dead-end.

All parameters are loaded from config/params.yaml via the launch file.
The instructor only needs to change params.yaml for different A, B, obstacles.
"""
import math
import os
import sys
import threading
import time
from typing import List, Tuple, Optional
import yaml

import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from sensor_msgs.msg import Range

# ── resolve our sibling package ──────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pa4_planner.occupancy_grid import OccupancyGrid
from pa4_planner.astar import astar, downsample_path
from pa4_planner.dwa import DWAPlanner
from pa4_planner.visualizer import Visualizer


# ── tiny helpers ─────────────────────────────────────────────────────────────

def _anorm(a: float) -> float:
    """Normalise angle to (-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _adiff(target: float, current: float) -> float:
    return _anorm(target - current)


# ── main node ────────────────────────────────────────────────────────────────

class PA4PlannerNode(DTROS):

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.PLANNING)

        # ── Load YAML config ─────────────────────────────────────────────────
        cfg_file = rospy.get_param("~config_file", "")
        if not cfg_file or not os.path.isfile(cfg_file):
            rospy.logfatal("[PA4] config_file param missing or file not found.")
            raise RuntimeError("config_file not set")

        with open(cfg_file) as f:
            self.cfg = yaml.safe_load(f)

        self._parse_cfg()

        # ── Occupancy grid ───────────────────────────────────────────────────
        m = self.cfg["map"]
        self.grid = OccupancyGrid(m["width"], m["height"], m["resolution"])

        # Planners
        self.dwa = DWAPlanner(self.cfg["dwa"])

        # Obstacle bookkeeping
        inflate = self.robot_radius + self.safety_margin
        self.all_obstacles: List[Tuple[float, float, float]] = []
        for ox, oy in self.cfg.get("known_obstacles", []):
            self.grid.add_obstacle(ox, oy, inflate)
            self.all_obstacles.append((ox, oy, inflate))

        # Unknown obstacles: stored separately; NOT in map until detected
        self.undiscovered: List[Tuple[float, float]] = list(
            self.cfg.get("unknown_obstacles", [])
        )

        # Odometry state
        self._lock = threading.Lock()
        sx, sy = self.start
        # Use configured / inferred starting heading so the world frame and
        # the robot's actual orientation agree from t=0. Without this the
        # planner assumes θ=0 and every subsequent leg is sheared by the
        # true initial heading.
        self.pose = [sx, sy, self.start_theta]
        rospy.loginfo(
            f"[PA4] Initial pose: ({sx:.2f}, {sy:.2f}, "
            f"{math.degrees(self.start_theta):.1f}°)"
        )
        self._left_prev: Optional[int] = None
        self._right_prev: Optional[int] = None
        self._left_delta: int = 0
        self._right_delta: int = 0
        self._v_curr: float = 0.0
        self._om_curr: float = 0.0

        # ToF sensor state
        # Latest validated forward distance reading (m). None ⇒ no recent / invalid.
        self._tof_lock = threading.Lock()
        self._tof_range: Optional[float] = None
        self._tof_last_stamp: Optional[rospy.Time] = None
        # Consecutive in-band readings observed by _sense_obstacles (hysteresis).
        self._tof_stable_count: int = 0

        # ROS interface
        veh = rospy.get_param("~veh", os.environ.get("VEHICLE_NAME", "duckiebot"))

        self._pub_wheels = rospy.Publisher(
            f"/{veh}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1,
        )
        self._sub_left = rospy.Subscriber(
            f"/{veh}/left_wheel_encoder_node/tick",
            WheelEncoderStamped,
            self._on_left_enc,
            queue_size=20,
        )
        self._sub_right = rospy.Subscriber(
            f"/{veh}/right_wheel_encoder_node/tick",
            WheelEncoderStamped,
            self._on_right_enc,
            queue_size=20,
        )

        # ToF subscriber (front-center range sensor)
        if self.tof_enabled:
            tof_topic = (
                self.tof_topic
                if self.tof_topic
                else f"/{veh}/front_center_tof_driver_node/range"
            )
            self._sub_tof = rospy.Subscriber(
                tof_topic,
                Range,
                self._on_tof,
                queue_size=1,
            )
            rospy.loginfo(f"[PA4] ToF subscribed: {tof_topic}")
        else:
            self._sub_tof = None
            rospy.loginfo("[PA4] ToF disabled in config.")

        # Visualizer
        self.viz: Optional[Visualizer] = None
        if self.cfg.get("visualization", {}).get("enabled", True):
            try:
                self.viz = Visualizer(self.grid, self.sensing_radius, self.robot_radius)
                self.viz.set_start_goal(self.start, self.goal)
            except Exception as exc:
                rospy.logwarn(f"[PA4] Visualizer unavailable: {exc}")

        rospy.loginfo("[PA4] Node initialised. Starting planner in 2 s …")
        rospy.sleep(2.0)  # let encoders settle

    # Config parsing

    def _parse_cfg(self) -> None:
        r = self.cfg["robot"]
        self.robot_radius = r["radius"]
        self.safety_margin = r["safety_margin"]
        self.baseline = r["baseline"]
        self.wheel_radius = r["wheel_radius"]
        self.ticks_per_rev = r["ticks_per_rev"]
        self.speed_gain = r["speed_gain"]          # m/s at command=1.0
        self.slow_speed = r["slow_speed"]           # m/s normal travel
        self.turn_speed_cmd = r["turn_speed_cmd"]   # normalised [-1,1]
        self.turn_tol = r["turn_tolerance"]
        self.wp_tol = r["wp_tolerance"]
        self.goal_tol = r["goal_tolerance"]
        self.wp_timeout = r["wp_timeout"]

        self.start: Tuple[float, float] = tuple(self.cfg["start"])
        self.goal:  Tuple[float, float] = tuple(self.cfg["goal"])

        # Initial heading (world frame, radians). Default: assume the robot
        # is placed pointing along the start→goal vector. This eliminates the
        # constant heading offset that otherwise rotates the entire path
        # (e.g. running at 30° when the actual diagonal is 45°).
        sx, sy = self.start
        gx, gy = self.goal
        default_theta = math.atan2(gy - sy, gx - sx)
        if "start_theta_deg" in self.cfg:
            self.start_theta = math.radians(float(self.cfg["start_theta_deg"]))
        elif "start_theta" in self.cfg:
            self.start_theta = float(self.cfg["start_theta"])
        else:
            self.start_theta = default_theta

        s = self.cfg["sensing"]
        self.sensing_radius = s["radius"]
        self.sensing_fov = math.radians(s["fov_deg"])

        tof = self.cfg.get("tof", {})
        self.tof_enabled = bool(tof.get("enabled", True))
        self.tof_topic = str(tof.get("topic", "") or "")
        self.tof_register_min = float(tof.get("register_min", 0.05))
        self.tof_register_max = float(tof.get("register_max", self.sensing_radius))
        self.tof_stable_readings = max(1, int(tof.get("stable_readings", 2)))
        self.tof_stale_timeout = float(tof.get("stale_timeout", 1.0))
        self.tof_dedup_radius = float(tof.get("dedup_radius", 0.12))
        # Forward offset of the ToF lens from the wheel-axle centre (robot origin).
        # On DB21M the front-center ToF sits roughly at the front bumper.
        self.tof_offset = float(tof.get("mount_offset", self.robot_radius))

        t = self.cfg["timing"]
        self.move_rate = t["move_rate"]
        self.sense_pause = t["sense_pause"]
        self.stop_hold_ticks = max(1, int(t.get("stop_hold_ticks", 3)))

        p = self.cfg["path"]
        self.path_min_spacing = p["min_spacing"]
        self.path_angle_thresh = p["angle_threshold"]
        self.path_max_drift = float(p.get("max_drift", 0.20))

        # Approach / deceleration tuning
        # When closer than approach_distance to the current waypoint, DWA's
        # max_v is interpolated linearly between min_approach_speed (at the
        # waypoint) and the regular max_v (at approach_distance).
        ap = self.cfg.get("approach", {})
        self.approach_distance = float(ap.get("distance", 0.25))
        self.approach_min_speed = float(ap.get("min_speed", 0.05))

        self.max_backtracks = int(self.cfg.get("max_backtracks", 5))

    # Encoder callbacks

    def _on_left_enc(self, msg: WheelEncoderStamped) -> None:
        with self._lock:
            if self._left_prev is None:
                self._left_prev = msg.data
            else:
                self._left_delta += msg.data - self._left_prev
                self._left_prev = msg.data

    def _on_right_enc(self, msg: WheelEncoderStamped) -> None:
        with self._lock:
            if self._right_prev is None:
                self._right_prev = msg.data
            else:
                self._right_delta += msg.data - self._right_prev
                self._right_prev = msg.data

    # ToF callback

    def _on_tof(self, msg: Range) -> None:
        """Cache the latest ToF reading; mark invalid out-of-range/NaN as None."""
        r = float(msg.range)
        valid = (
            math.isfinite(r)
            and msg.min_range <= r <= msg.max_range
        )
        with self._tof_lock:
            self._tof_range = r if valid else None
            self._tof_last_stamp = (
                msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
            )

    def _get_tof_distance(self) -> Optional[float]:
        """
        Return ToF distance (m) only if it has been observed in the registration
        band for `tof_stable_readings` consecutive sense ticks AND is recent.
        Otherwise returns None and resets the stability counter.
        """
        with self._tof_lock:
            d = self._tof_range
            stamp = self._tof_last_stamp

        if d is None or stamp is None:
            self._tof_stable_count = 0
            return None

        if (rospy.Time.now() - stamp).to_sec() > self.tof_stale_timeout:
            self._tof_stable_count = 0
            return None

        if self.tof_register_min <= d <= self.tof_register_max:
            self._tof_stable_count += 1
        else:
            self._tof_stable_count = 0

        if self._tof_stable_count >= self.tof_stable_readings:
            return d
        return None

    def _is_inside_map(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.grid.width_m and 0.0 <= y <= self.grid.height_m

    def _is_near_known_obstacle(self, x: float, y: float) -> bool:
        return any(
            math.hypot(x - ox, y - oy) < self.tof_dedup_radius
            for ox, oy, _ in self.all_obstacles
        )

    # Odometry

    def _process_odom(self) -> None:
        """Consume accumulated tick deltas and update pose. Call each control step."""
        with self._lock:
            dl_ticks = self._left_delta
            dr_ticks = self._right_delta
            self._left_delta = 0
            self._right_delta = 0

        dist_per_tick = (2.0 * math.pi * self.wheel_radius) / self.ticks_per_rev
        dl = dl_ticks * dist_per_tick   # metres left wheel
        dr = dr_ticks * dist_per_tick   # metres right wheel
        d = (dl + dr) / 2.0
        dtheta = (dr - dl) / self.baseline

        with self._lock:
            mid_th = self.pose[2] + dtheta / 2.0
            self.pose[0] += d * math.cos(mid_th)
            self.pose[1] += d * math.sin(mid_th)
            self.pose[2] = _anorm(self.pose[2] + dtheta)

    def _get_pose(self) -> Tuple[float, float, float]:
        with self._lock:
            return (self.pose[0], self.pose[1], self.pose[2])

    # Wheel commands

    def _send_cmd(self, v: float, omega: float) -> None:
        """Publish (v m/s, omega rad/s) as normalised wheel commands."""
        L = self.baseline
        v_r = v + omega * L / 2.0
        v_l = v - omega * L / 2.0
        gain = max(self.speed_gain, 1e-6)
        cmd_l = float(np.clip(v_l / gain, -1.0, 1.0))
        cmd_r = float(np.clip(v_r / gain, -1.0, 1.0))

        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left  = cmd_l
        msg.vel_right = cmd_r
        self._pub_wheels.publish(msg)

        with self._lock:
            self._v_curr  = v
            self._om_curr = omega

    def _stop(self) -> None:
        """
        Brake. Publishes a zero wheel command and HOLDS it for
        ``stop_hold_ticks`` consecutive sends so the driver actually
        decelerates the motors (Duckiebot motor latency is non-zero).
        """
        for _ in range(self.stop_hold_ticks):
            self._send_cmd(0.0, 0.0)
            rospy.sleep(1.0 / max(self.move_rate, 1))
        with self._lock:
            self._v_curr  = 0.0
            self._om_curr = 0.0

    # A* global planner

    def _plan_astar(
        self, from_pose: Optional[Tuple[float, float, float]] = None
    ) -> Optional[List[Tuple[float, float]]]:
        start = (from_pose[0], from_pose[1]) if from_pose else self.start
        raw = astar(self.grid, start, self.goal)
        if raw is None:
            return None
        path = downsample_path(raw, self.path_min_spacing, self.path_angle_thresh)
        rospy.loginfo(f"[PA4] A* path: {len(raw)} cells → {len(path)} waypoints")
        return path

    # Per-node actions

    def _turn_to(
        self,
        target: Tuple[float, float],
        rate: rospy.Rate,
    ) -> None:
        """Rotate in-place until heading matches direction to target."""
        omega_base = (2.0 * self.turn_speed_cmd * self.speed_gain) / self.baseline

        x0, y0, _ = self._get_pose()
        target_angle = math.atan2(target[1] - y0, target[0] - x0)
        rospy.loginfo(f"[PA4] Turning to {math.degrees(target_angle):.1f}° …")

        max_iters = int(30 * self.move_rate)   # 30 s hard timeout
        for _ in range(max_iters):
            if rospy.is_shutdown():
                break
            self._process_odom()
            x, y, theta = self._get_pose()
            target_angle = math.atan2(target[1] - y, target[0] - x)
            err = _adiff(target_angle, theta)

            if abs(err) < self.turn_tol:
                self._stop()
                return

            omega = omega_base * math.copysign(1.0, err)
            self._send_cmd(0.0, omega)
            self._update_viz(self._get_pose(), None, None, None)
            rate.sleep()

        rospy.logwarn("[PA4] Turn timeout — proceeding with current heading.")
        self._stop()

    def _sense_obstacles(self) -> List[Tuple[float, float]]:
        """
        Detect new (previously unknown) obstacles from two sources:

        1. Real ToF sensor — a single front-facing range beam. If the latest
           reading lies in the registration band and stays there for
           ``tof.stable_readings`` consecutive sense ticks, an obstacle is
           registered at ``(x + d·cosθ, y + d·sinθ)`` in the world frame.
        2. Simulated ``unknown_obstacles`` list — used for offline testing.
           Any entry within ``sensing.radius`` AND inside the forward FOV
           cone is treated as just-detected.

        Returns world-frame (x, y) coordinates of newly detected obstacles.
        Duplicates (already in the map) are filtered out.
        """
        x, y, theta = self._get_pose()
        detected: List[Tuple[float, float]] = []

        # Real ToF reading
        if self.tof_enabled:
            tof_d = self._get_tof_distance()
            if tof_d is not None:
                # Beam originates at the front-mounted ToF lens, not the
                # robot's wheel-axle centre. Project the hit point from
                # (sensor_x, sensor_y) along the robot heading by `tof_d`.
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                sxp = x + self.tof_offset * cos_t
                syp = y + self.tof_offset * sin_t
                ox = sxp + tof_d * cos_t
                oy = syp + tof_d * sin_t

                # Geometric sanity: refuse to register a hit whose inflated
                # footprint would engulf the robot itself. Such "obstacles"
                # only ever cause the infinite blocked-replan loop because
                # A* will never find a path that starts outside the halo.
                # Require centre-to-centre distance ≥ inflate + a small
                # margin (one cell). Otherwise: ignore the reading.
                inflate = self.robot_radius + self.safety_margin
                min_stand_off = inflate + self.grid.res
                d_from_robot = math.hypot(ox - x, oy - y)

                if d_from_robot < min_stand_off:
                    rospy.logwarn(
                        f"[PA4] ToF hit {tof_d:.2f} m → world "
                        f"({ox:.2f},{oy:.2f}) is inside robot footprint "
                        f"({d_from_robot:.2f} m < {min_stand_off:.2f} m). "
                        f"Ignored — sensor noise or robot is touching wall."
                    )
                elif self._is_inside_map(ox, oy) and not self._is_near_known_obstacle(ox, oy):
                    rospy.loginfo(
                        f"[PA4] ToF hit at d={tof_d:.2f} m → "
                        f"obstacle ({ox:.2f}, {oy:.2f})"
                    )
                    detected.append((ox, oy))

        # Simulated ``unknown_obstacles`` (offline / testing)
        remaining = []
        for ox, oy in self.undiscovered:
            dist = math.hypot(x - ox, y - oy)
            if dist <= self.sensing_radius:
                angle_to = math.atan2(oy - y, ox - x)
                diff = abs(_adiff(angle_to, theta))
                if diff <= self.sensing_fov / 2.0:
                    detected.append((ox, oy))
                    continue
            remaining.append((ox, oy))
        self.undiscovered = remaining

        return detected

    def _path_drift(
        self,
        pose: Tuple[float, float, float],
        path: List[Tuple[float, float]],
    ) -> float:
        """
        Shortest perpendicular distance from the robot to the polyline
        ``path``. Returns ``+inf`` for empty / single-point paths.
        Used to decide when to abandon the current A* plan and replan
        from the live pose rather than dragging the robot back onto an
        old line that no longer reflects the current geometry.
        """
        if not path or len(path) < 2:
            return float("inf")
        px, py = pose[0], pose[1]
        best = float("inf")
        for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-9:
                d = math.hypot(px - ax, py - ay)
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
                t = max(0.0, min(1.0, t))
                cx, cy = ax + t * dx, ay + t * dy
                d = math.hypot(px - cx, py - cy)
            if d < best:
                best = d
        return best

    def _direction_blocked(
        self,
        pose: Tuple[float, float, float],
        target: Tuple[float, float],
    ) -> bool:
        """
        True only if the straight segment pose→target passes through an
        OCCUPIED cell (a real obstacle). INFLATED cells are NOT treated as
        a hard block: they're a soft buffer the DWA cost handles. The
        first ``robot.radius`` metres of the segment are also skipped so
        the robot can move out of its own inflation halo right after a
        close ToF registration (was the source of the infinite
        "blocked → replan" loop).
        """
        return self.grid.hard_blocks_segment(
            pose[0], pose[1], target[0], target[1],
            ignore_radius_m=self.robot_radius,
        )

    def _move_to_waypoint(
        self,
        target: Tuple[float, float],
        global_path: List[Tuple[float, float]],
        rate: rospy.Rate,
    ) -> bool:
        """
        DWA-guided motion toward a single waypoint.

        Key behaviours that keep the trajectory clean between waypoints:

        * DWA receives ONLY the current waypoint as its path reference, never
          the future ones — otherwise the path-following term curves the
          robot around the current node toward the next leg (the "C-shape").
        * As the robot approaches the waypoint (``dist < approach_distance``)
          DWA's max linear velocity is linearly scaled down toward
          ``approach_min_speed``. The robot decelerates instead of barrelling
          through and overshooting.
        * ``self.dwa.plan(...)`` is given the unicycle state from odometry
          and the latest current_v / current_om for a tight dynamic window.

        Returns True when the waypoint is reached, False on timeout, total
        DWA failure, or a newly detected obstacle (replan in caller).

        ``global_path`` is accepted only for visualisation; the planner
        substitutes ``[target]`` when calling DWA.
        """
        deadline = rospy.Time.now() + rospy.Duration(self.wp_timeout)
        consecutive_zero = 0

        # DWA reference path: just the current target. Prevents the cost
        # function from biasing the trajectory toward future waypoints.
        dwa_ref_path = [target]

        while not rospy.is_shutdown():
            self._process_odom()
            x, y, theta = self._get_pose()

            dist = math.hypot(x - target[0], y - target[1])
            if dist < self.wp_tol:
                self._stop()
                return True

            if rospy.Time.now() > deadline:
                self._stop()
                rospy.logwarn("[PA4] Waypoint timeout.")
                return False

            new_obs = self._sense_obstacles()
            if new_obs:
                self._stop()
                self._register_new_obstacles(new_obs)
                return False  # triggers REPLAN in caller

            # Drift check: if the robot has wandered far from the global
            # polyline (e.g. avoided an obstacle and is now well off the
            # original A* line), don't try to "snake back" onto the stale
            # path — bail out so the caller replans from the live pose.
            if (
                self.path_max_drift > 0.0
                and len(global_path) >= 2
                and self._path_drift((x, y, theta), global_path) > self.path_max_drift
            ):
                self._stop()
                rospy.logwarn(
                    "[PA4] Off-path drift exceeded — replanning from current pose."
                )
                return False

            with self._lock:
                v_c  = self._v_curr
                om_c = self._om_curr

            # Decelerate as we get close to the waypoint to prevent overshoot.
            if dist < self.approach_distance:
                t = max(0.0, dist - self.wp_tol) / max(
                    self.approach_distance - self.wp_tol, 1e-3
                )
                max_v_eff = (
                    self.approach_min_speed
                    + t * (self.slow_speed - self.approach_min_speed)
                )
            else:
                max_v_eff = self.slow_speed

            best_v, best_om, all_trajs, best_traj = self.dwa.plan(
                state=(x, y, theta),
                goal=target,
                global_path=dwa_ref_path,
                obstacles=self.all_obstacles,
                current_v=v_c,
                current_om=om_c,
                max_v_override=max_v_eff,
            )

            if best_v == 0.0 and best_om == 0.0:
                consecutive_zero += 1
                if consecutive_zero > self.move_rate * 3:  # 3 s stuck
                    self._stop()
                    rospy.logwarn("[PA4] DWA stuck for 3 s — triggering replan.")
                    return False
            else:
                consecutive_zero = 0

            self._send_cmd(best_v, best_om)
            # Visualise the original (full) global path so the operator can
            # still see where the robot is heading after this waypoint.
            self._update_viz((x, y, theta), global_path, all_trajs, best_traj)
            rate.sleep()

        return False

    def _register_new_obstacles(
        self, detected: List[Tuple[float, float]]
    ) -> None:
        inflate = self.robot_radius + self.safety_margin
        for ox, oy in detected:
            rospy.loginfo(f"[PA4] NEW obstacle detected at ({ox:.2f}, {oy:.2f})")
            self.grid.add_obstacle(ox, oy, inflate)
            self.all_obstacles.append((ox, oy, inflate))

    def _backtrack(
        self,
        history: List[Tuple[float, float, float]],
        rate: rospy.Rate,
    ) -> Optional[List[Tuple[float, float]]]:
        """Move to the previous visited pose and replan. Returns new path or None."""
        if not history:
            return None
        prev_pose = history.pop()
        rospy.logwarn(
            f"[PA4] Backtracking to ({prev_pose[0]:.2f}, {prev_pose[1]:.2f})"
        )
        # Move there using DWA (treat prev_pose as new temporary goal)
        prev_goal = (prev_pose[0], prev_pose[1])
        self._move_to_waypoint(prev_goal, [prev_goal], rate)
        new_path = self._plan_astar(from_pose=self._get_pose())
        return new_path

    # ── Visualisation helper ──────────────────────────────────────────────────

    def _update_viz(
        self,
        pose: Tuple[float, float, float],
        path: Optional[list],
        dwa_trajs: Optional[list],
        best_traj: Optional[list],
    ) -> None:
        if self.viz is None:
            return
        try:
            self.viz.update(
                robot_pose=pose,
                global_path=path or [],
                dwa_trajs=dwa_trajs or [],
                best_traj=best_traj or [],
                obstacles=self.all_obstacles,
            )
        except Exception:
            pass  # never crash the planner due to viz error

    # Main state machine

    def run(self) -> None:
        rate = rospy.Rate(self.move_rate)

        # Initial plan
        waypoints = self._plan_astar()
        if waypoints is None:
            rospy.logerr("[PA4] No path A→B at startup. Check obstacle layout.")
            self._stop()
            return

        wp_idx = 1       # path[0] is start position — already there, skip it
        history: List[Tuple[float, float, float]] = []
        backtrack_count = 0
        active_path = waypoints

        # Loop guard: count consecutive replans without any meaningful
        # motion. If we trip the "Direct path blocked" branch more than a
        # few times without moving, abort instead of spinning forever.
        stuck_replans = 0
        last_motion_pose = self._get_pose()

        rospy.loginfo("[PA4] Starting traversal …")

        while not rospy.is_shutdown():
            self._process_odom()
            pose = self._get_pose()

            # Goal check
            if math.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1]) < self.goal_tol:
                self._stop()
                rospy.loginfo("[PA4] ✓ Goal reached!")
                self._update_viz(pose, active_path, None, None)
                break

            # Waypoints exhausted without reaching goal
            if wp_idx >= len(active_path):
                rospy.logwarn("[PA4] Waypoints exhausted — replanning.")
                new_path = self._plan_astar(from_pose=pose)
                if new_path is None:
                    new_path = self._backtrack(history, rate)
                    backtrack_count += 1
                if new_path is None or backtrack_count > self.max_backtracks:
                    rospy.logerr("[PA4] Dead end: cannot reach goal. Stopping.")
                    self._stop()
                    break
                active_path = new_path
                wp_idx = 1   # new path[0] = current pos, skip it
                continue

            target = active_path[wp_idx]

            # ── Drift check ─────────────────────────────────────────────────
            # If the robot is significantly off the current A* polyline,
            # discard the stale plan and replan from the live pose rather
            # than trying to crawl back onto a path that no longer makes
            # sense (e.g. after a detour around a freshly registered
            # obstacle).
            if (
                self.path_max_drift > 0.0
                and self._path_drift(pose, active_path) > self.path_max_drift
            ):
                rospy.logwarn(
                    "[PA4] Drifted off A* path — replanning from current pose."
                )
                new_path = self._plan_astar(from_pose=pose)
                if new_path is None:
                    new_path = self._backtrack(history, rate)
                    backtrack_count += 1
                if new_path is None or backtrack_count > self.max_backtracks:
                    rospy.logerr("[PA4] Dead end after drift replan.")
                    self._stop()
                    break
                active_path = new_path
                wp_idx = 1
                last_motion_pose = self._get_pose()
                stuck_replans = 0
                continue

            # Step 1: TURN to face next waypoint.
            #         Skip if direction is blocked by a known obstacle → replan.
            if self._direction_blocked(pose, target):
                # Did we make any real motion since the last replan?
                moved = math.hypot(
                    pose[0] - last_motion_pose[0],
                    pose[1] - last_motion_pose[1],
                ) > max(self.wp_tol * 0.5, 0.03)
                if moved:
                    stuck_replans = 0
                    last_motion_pose = pose
                else:
                    stuck_replans += 1

                rospy.logwarn(
                    f"[PA4] Direct path to waypoint blocked — replanning "
                    f"(stuck_replans={stuck_replans})."
                )

                if stuck_replans > 3:
                    rospy.logerr(
                        "[PA4] Replanning loop without motion. Forcing "
                        "backtrack."
                    )
                    new_path = self._backtrack(history, rate)
                    backtrack_count += 1
                    stuck_replans = 0
                    last_motion_pose = self._get_pose()
                else:
                    new_path = self._plan_astar(from_pose=pose)
                    if new_path is None:
                        new_path = self._backtrack(history, rate)
                        backtrack_count += 1
                        last_motion_pose = self._get_pose()

                if new_path is None or backtrack_count > self.max_backtracks:
                    rospy.logerr("[PA4] Dead end after block detection.")
                    self._stop()
                    break
                active_path = new_path
                wp_idx = 1   # new path[0] = current pos, skip it
                continue

            self._turn_to(target, rate)

            # ─────────────────────────────────────────────────────────────────
            # Step 2: SENSE in the forward direction.
            # Poll _sense_obstacles() throughout sense_pause so the ToF stable
            # counter has multiple ticks to accumulate — a single call after
            # sleep(sense_pause) cannot reach stable_readings > 1.
            # ─────────────────────────────────────────────────────────────────
            rospy.loginfo(f"[PA4] Sensing at waypoint {wp_idx} …")
            self._update_viz(self._get_pose(), active_path, None, None)
            new_obs: List[Tuple[float, float]] = []
            sense_rate = rospy.Rate(self.move_rate)
            sense_deadline = rospy.Time.now() + rospy.Duration(self.sense_pause)
            while rospy.Time.now() < sense_deadline:
                hits = self._sense_obstacles()
                if hits:
                    new_obs = hits
                    break
                sense_rate.sleep()

            # ─────────────────────────────────────────────────────────────────
            # Step 3: If obstacle detected → update map → REPLAN.
            # ─────────────────────────────────────────────────────────────────
            if new_obs:
                self._register_new_obstacles(new_obs)
                new_path = self._plan_astar(from_pose=self._get_pose())
                if new_path is None:
                    new_path = self._backtrack(history, rate)
                    backtrack_count += 1
                if new_path is None or backtrack_count > self.max_backtracks:
                    rospy.logerr("[PA4] Dead end after obstacle discovery.")
                    self._stop()
                    break
                active_path = new_path
                wp_idx = 1   # new path[0] = current pos, skip it
                continue

            # ─────────────────────────────────────────────────────────────────
            # Step 4 / 5: No obstacle → MOVE via DWA.
            # ─────────────────────────────────────────────────────────────────
            rospy.loginfo(
                f"[PA4] Moving to wp {wp_idx}: ({target[0]:.2f}, {target[1]:.2f})"
            )
            reached = self._move_to_waypoint(target, active_path[wp_idx:], rate)

            if reached:
                history.append(self._get_pose())
                backtrack_count = 0
                stuck_replans = 0
                last_motion_pose = self._get_pose()
                wp_idx += 1
            else:
                # DWA failed or new obstacle found during movement
                pose = self._get_pose()
                new_path = self._plan_astar(from_pose=pose)
                if new_path is None:
                    new_path = self._backtrack(history, rate)
                    backtrack_count += 1
                if new_path is None or backtrack_count > self.max_backtracks:
                    rospy.logerr("[PA4] Dead end: DWA failed, no alternative path.")
                    self._stop()
                    break
                active_path = new_path
                wp_idx = 1   # new path[0] = current pos, skip it

        if self.viz:
            self.viz.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = PA4PlannerNode(node_name="pa4_planner_node")
        node.run()
    except rospy.ROSInterruptException:
        pass
