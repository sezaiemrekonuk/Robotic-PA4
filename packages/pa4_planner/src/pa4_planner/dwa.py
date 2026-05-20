"""Dynamic Window Approach (DWA) local planner for differential-drive robot."""
import math
from typing import List, Optional, Tuple
import numpy as np


class DWAPlanner:
    """
    Samples (v, ω) pairs inside the dynamic window, simulates each trajectory,
    scores them, and returns the best control.

    Cost = w_goal  * dist_to_goal
         + w_path  * dist_to_global_path
         + w_obs   * max(0, clearance_threshold − min_clearance)  ← penalty only when close
         - w_vel   * normalised_forward_velocity  ← reward for moving forward

    Any trajectory that enters an inflated obstacle is rejected.
    """

    def __init__(self, cfg: dict):
        self.dt = cfg["dt"]
        self.n_steps = max(1, int(cfg["horizon"] / cfg["dt"]))
        self.v_samples = cfg["v_samples"]
        self.om_samples = cfg["omega_samples"]
        self.max_v = cfg["max_v"]
        self.min_v = cfg["min_v"]
        self.max_om = cfg["max_omega"]
        self.max_acc = cfg["max_accel"]
        self.max_dyaw = cfg["max_dyaw"]
        self.w_goal = cfg["weight_goal"]
        self.w_path = cfg["weight_path"]
        self.w_obs = cfg["weight_obstacle"]
        self.w_vel = cfg["weight_velocity"]
        # Only penalise obstacle proximity below this distance; no reward for being far.
        self.clearance_threshold = cfg.get("clearance_threshold", 0.30)

    # ── Public API ───────────────────────────────────────────────────────────

    def plan(
        self,
        state: Tuple[float, float, float],
        goal: Tuple[float, float],
        global_path: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
        current_v: float = 0.0,
        current_om: float = 0.0,
        max_v_override: Optional[float] = None,
    ) -> Tuple[float, float, list, list]:
        """
        Returns (best_v, best_omega, all_trajectories, best_trajectory).
        all_trajectories: list of (traj_points, is_valid) for visualisation.

        ``max_v_override``: when set, clips the upper bound of the linear
        velocity dynamic window. Used by the caller to decelerate near the
        waypoint and prevent overshoot / curving past the target.
        """
        v_hi_cap = self.max_v if max_v_override is None else min(self.max_v, max_v_override)
        v_win = self._window(current_v, self.max_acc, self.min_v, v_hi_cap)
        om_win = self._window(current_om, self.max_dyaw, -self.max_om, self.max_om)

        v_range = np.linspace(v_win[0], v_win[1], self.v_samples)
        om_range = np.linspace(om_win[0], om_win[1], self.om_samples)

        best_score = -float("inf")
        best_v, best_om = 0.0, 0.0
        best_traj: List[Tuple[float, float]] = [state[:2]]
        all_trajs: List[Tuple[list, bool]] = []

        for v in v_range:
            for om in om_range:
                traj = self._simulate(state, v, om)
                valid, score = self._score(traj, goal, global_path, obstacles, v)
                all_trajs.append((traj, valid))
                if valid and score > best_score:
                    best_score = score
                    best_v, best_om = v, om
                    best_traj = traj

        if best_score == -float("inf"):
            # All trajectories blocked — stop
            best_v, best_om = 0.0, 0.0
            best_traj = [state[:2]]

        return best_v, best_om, all_trajs, best_traj

    # ── Internals ────────────────────────────────────────────────────────────

    def _window(self, cur: float, acc: float, lo: float, hi: float) -> Tuple:
        w_lo = max(lo, cur - acc * self.dt)
        w_hi = min(hi, cur + acc * self.dt)
        if w_lo > w_hi:
            w_lo = w_hi = max(lo, min(hi, cur))
        return (w_lo, w_hi)

    def _simulate(
        self, state: Tuple[float, float, float], v: float, om: float
    ) -> List[Tuple[float, float]]:
        x, y, th = state
        pts = [(x, y)]
        for _ in range(self.n_steps):
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            th += om * self.dt
            pts.append((x, y))
        return pts

    def _score(
        self,
        traj: List[Tuple[float, float]],
        goal: Tuple[float, float],
        global_path: List[Tuple[float, float]],
        obstacles: List[Tuple[float, float, float]],
        v: float,
    ) -> Tuple[bool, float]:
        # Collision check — whole trajectory
        for px, py in traj:
            for ox, oy, ir in obstacles:
                if math.hypot(px - ox, py - oy) < ir:
                    return False, -float("inf")

        end_x, end_y = traj[-1]

        # Distance to goal (end of trajectory)
        goal_dist = math.hypot(end_x - goal[0], end_y - goal[1])

        # Min distance from endpoint to global path points
        if global_path:
            path_dist = min(math.hypot(end_x - px, end_y - py) for px, py in global_path)
        else:
            path_dist = 0.0

        # Obstacle penalty: only fires when clearance < threshold.
        # Does NOT reward being far away (prevents staying-still bias).
        if obstacles:
            min_clear = min(
                math.hypot(px - ox, py - oy) - ir
                for px, py in traj
                for ox, oy, ir in obstacles
            )
            obs_penalty = max(0.0, self.clearance_threshold - min_clear)
        else:
            obs_penalty = 0.0

        score = (
            -self.w_goal * goal_dist
            - self.w_path * path_dist
            - self.w_obs  * obs_penalty
            + self.w_vel  * (v / max(self.max_v, 1e-6))
        )
        return True, score
