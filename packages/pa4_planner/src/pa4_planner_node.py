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

        # ── Planners ─────────────────────────────────────────────────────────
        self.dwa = DWAPlanner(self.cfg["dwa"])

        # ── Obstacle bookkeeping ─────────────────────────────────────────────
        inflate = self.robot_radius + self.safety_margin
        self.all_obstacles: List[Tuple[float, float, float]] = []
        for ox, oy in self.cfg.get("known_obstacles", []):
            self.grid.add_obstacle(ox, oy, inflate)
            self.all_obstacles.append((ox, oy, inflate))

        # Unknown obstacles: stored separately; NOT in map until detected
        self.undiscovered: List[Tuple[float, float]] = list(
            self.cfg.get("unknown_obstacles", [])
        )

        # ── Odometry state ───────────────────────────────────────────────────
        self._lock = threading.Lock()
        sx, sy = self.start
        self.pose = [sx, sy, 0.0]   # [x, y, theta] — world frame, metres/radians
        self._left_prev: Optional[int] = None
        self._right_prev: Optional[int] = None
        self._left_delta: int = 0
        self._right_delta: int = 0
        self._v_curr: float = 0.0
        self._om_curr: float = 0.0

        # ── ROS interface ────────────────────────────────────────────────────
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

        # ── Visualizer ───────────────────────────────────────────────────────
        self.viz: Optional[Visualizer] = None
        if self.cfg.get("visualization", {}).get("enabled", True):
            try:
                self.viz = Visualizer(self.grid, self.sensing_radius, self.robot_radius)
                self.viz.set_start_goal(self.start, self.goal)
            except Exception as exc:
                rospy.logwarn(f"[PA4] Visualizer unavailable: {exc}")

        rospy.loginfo("[PA4] Node initialised. Starting planner in 2 s …")
        rospy.sleep(2.0)  # let encoders settle

    # ── Config parsing ────────────────────────────────────────────────────────

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

        s = self.cfg["sensing"]
        self.sensing_radius = s["radius"]
        self.sensing_fov = math.radians(s["fov_deg"])

        t = self.cfg["timing"]
        self.move_rate = t["move_rate"]
        self.sense_pause = t["sense_pause"]

        p = self.cfg["path"]
        self.path_min_spacing = p["min_spacing"]
        self.path_angle_thresh = p["angle_threshold"]

        self.max_backtracks = int(self.cfg.get("max_backtracks", 5))

    # ── Encoder callbacks ─────────────────────────────────────────────────────

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

    # ── Odometry ─────────────────────────────────────────────────────────────

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

    # ── Wheel commands ────────────────────────────────────────────────────────

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
        self._send_cmd(0.0, 0.0)
        with self._lock:
            self._v_curr  = 0.0
            self._om_curr = 0.0

    # ── A* global planner ─────────────────────────────────────────────────────

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

    # ── Per-node actions ──────────────────────────────────────────────────────

    def _turn_to(
        self,
        target: Tuple[float, float],
        rate: rospy.Rate,
    ) -> None:
        """Rotate in-place until heading matches direction to target."""
        # omega_base [rad/s]: derived from normalised wheel cmd so that
        # v_r = +turn_speed_cmd * speed_gain  and  v_l = -turn_speed_cmd * speed_gain
        # ⟹  omega = (v_r − v_l) / baseline = 2 × cmd × gain / baseline
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
        Simulates on-board sensor: any undiscovered obstacle within sensing_radius
        AND inside the field-of-view cone is reported as newly detected.
        In the real-robot case this would use camera / ToF data.
        """
        x, y, theta = self._get_pose()
        detected = []
        remaining = []

        for ox, oy in self.undiscovered:
            dist = math.hypot(x - ox, y - oy)
            if dist <= self.sensing_radius:
                # Check inside FOV cone (centred on current heading)
                angle_to = math.atan2(oy - y, ox - x)
                diff = abs(_adiff(angle_to, theta))
                if diff <= self.sensing_fov / 2.0:
                    detected.append((ox, oy))
                    continue
            remaining.append((ox, oy))

        self.undiscovered = remaining
        return detected

    def _direction_blocked(
        self,
        pose: Tuple[float, float, float],
        target: Tuple[float, float],
    ) -> bool:
        """True if the straight segment pose→target passes through a known obstacle."""
        return self.grid.collides_segment(pose[0], pose[1], target[0], target[1])

    def _move_to_waypoint(
        self,
        target: Tuple[float, float],
        global_path: List[Tuple[float, float]],
        rate: rospy.Rate,
    ) -> bool:
        """
        DWA-guided motion toward target.
        Returns True when waypoint reached, False on timeout or total DWA failure.
        Also checks for undiscovered obstacles during movement.
        """
        deadline = rospy.Time.now() + rospy.Duration(self.wp_timeout)
        consecutive_zero = 0

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

            # Also sense during motion (real-time unknown obstacle detection)
            new_obs = self._sense_obstacles()
            if new_obs:
                self._stop()
                self._register_new_obstacles(new_obs)
                return False  # triggers REPLAN in caller

            with self._lock:
                v_c  = self._v_curr
                om_c = self._om_curr

            best_v, best_om, all_trajs, best_traj = self.dwa.plan(
                state=(x, y, theta),
                goal=target,
                global_path=global_path,
                obstacles=self.all_obstacles,
                current_v=v_c,
                current_om=om_c,
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

    # ── Main state machine ────────────────────────────────────────────────────

    def run(self) -> None:
        rate = rospy.Rate(self.move_rate)

        # ── Initial plan ─────────────────────────────────────────────────────
        waypoints = self._plan_astar()
        if waypoints is None:
            rospy.logerr("[PA4] No path A→B at startup. Check obstacle layout.")
            self._stop()
            return

        wp_idx = 1       # path[0] is start position — already there, skip it
        history: List[Tuple[float, float, float]] = []
        backtrack_count = 0
        active_path = waypoints

        rospy.loginfo("[PA4] Starting traversal …")

        while not rospy.is_shutdown():
            self._process_odom()
            pose = self._get_pose()

            # ── Goal check ───────────────────────────────────────────────────
            if math.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1]) < self.goal_tol:
                self._stop()
                rospy.loginfo("[PA4] ✓ Goal reached!")
                self._update_viz(pose, active_path, None, None)
                break

            # ── Waypoints exhausted without reaching goal ─────────────────
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

            # ─────────────────────────────────────────────────────────────────
            # Step 1: TURN to face next waypoint.
            #         Skip if direction is blocked by a known obstacle → replan.
            # ─────────────────────────────────────────────────────────────────
            if self._direction_blocked(pose, target):
                rospy.logwarn("[PA4] Direct path to waypoint blocked — replanning.")
                new_path = self._plan_astar(from_pose=pose)
                if new_path is None:
                    new_path = self._backtrack(history, rate)
                    backtrack_count += 1
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
            # ─────────────────────────────────────────────────────────────────
            rospy.loginfo(f"[PA4] Sensing at waypoint {wp_idx} …")
            self._update_viz(self._get_pose(), active_path, None, None)
            rospy.sleep(self.sense_pause)

            new_obs = self._sense_obstacles()

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
