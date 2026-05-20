"""Real-time matplotlib visualizer for PA4 path planner."""
import math
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .occupancy_grid import OccupancyGrid

# Try hardware backends, fall back to Agg (no display)
for _backend in ("TkAgg", "Qt5Agg", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue


class Visualizer:
    """
    Draws — all simultaneously — on a single matplotlib figure:
      • Occupancy grid (updating as obstacles are discovered)
      • Global A* path
      • Robot body + heading arrow + sensing circle
      • DWA candidate trajectories (gray=valid, salmon=rejected)
      • DWA best trajectory (cyan, thick)
      • Each obstacle body (black) + inflated boundary (dashed red)
      • Odometry trail (faint green)
      • Start (green triangle) and Goal (gold star)
    """

    def __init__(self, grid: OccupancyGrid, sensing_radius: float, robot_radius: float):
        self.grid = grid
        self.sensing_radius = sensing_radius
        self.robot_radius = robot_radius
        self._headless = matplotlib.get_backend() == "Agg"

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("PA4 — A* + DWA Path Planner")
        self._build_static_artists()
        self._dwa_lines: list = []
        self._obs_patches: list = []
        self._odom_xs: List[float] = []
        self._odom_ys: List[float] = []

    # ── Initial canvas ───────────────────────────────────────────────────────

    def _build_static_artists(self) -> None:
        ax = self.ax
        ax.set_xlim(0.0, self.grid.width_m)
        ax.set_ylim(0.0, self.grid.height_m)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_title("PA4: A* Global Path + DWA Local Planner", fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        # Grid background — free=white, inflated=orange, occupied=black
        self._grid_img = ax.imshow(
            np.zeros((self.grid.rows, self.grid.cols)),
            origin="lower",
            extent=[0.0, self.grid.width_m, 0.0, self.grid.height_m],
            cmap="RdYlGn_r",
            vmin=0, vmax=2,
            alpha=0.45,
            zorder=0,
        )

        # Global A* path
        (self._path_line,) = ax.plot([], [], "b-", lw=2.0, label="A* Path", zorder=3)

        # DWA best trajectory
        (self._best_traj,) = ax.plot([], [], color="cyan", lw=2.5, label="DWA Best", zorder=5)

        # Odometry trail
        (self._odom_line,) = ax.plot([], [], color="limegreen", lw=1.0,
                                      alpha=0.5, label="Odometry Trail", zorder=2)

        # Robot body
        self._robot_body = plt.Circle(
            (0, 0), self.robot_radius, color="red", zorder=6, label="Robot"
        )
        ax.add_patch(self._robot_body)

        # Heading arrow — placeholder; recreated each update
        self._arrow = ax.annotate(
            "", xy=(0.1, 0.1), xytext=(0.05, 0.05),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=2.0),
            zorder=7,
        )

        # Sensing circle
        self._sense_circle = plt.Circle(
            (0, 0), self.sensing_radius,
            color="gold", fill=False, linestyle="--", lw=1.5,
            label="Sensing Area", zorder=4,
        )
        ax.add_patch(self._sense_circle)

        # Start / Goal markers (set later via set_start_goal)
        (self._start_mark,) = ax.plot([], [], "g^", ms=12, zorder=8, label="Start")
        (self._goal_mark,)  = ax.plot([], [], "g*", ms=16, zorder=8, label="Goal")

        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        plt.tight_layout()

    # ── Public methods ───────────────────────────────────────────────────────

    def set_start_goal(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> None:
        self._start_mark.set_data([start[0]], [start[1]])
        self._goal_mark.set_data([goal[0]], [goal[1]])
        self._flush()

    def update(
        self,
        robot_pose: Tuple[float, float, float],
        global_path: Optional[List[Tuple[float, float]]],
        dwa_trajs: Optional[List[Tuple[list, bool]]],
        best_traj: Optional[List[Tuple[float, float]]],
        obstacles: List[Tuple[float, float, float]],
    ) -> None:
        x, y, theta = robot_pose

        # Grid background
        self._grid_img.set_data(self.grid.get_grid())

        # Global path
        if global_path and len(global_path) >= 2:
            self._path_line.set_data(
                [p[0] for p in global_path],
                [p[1] for p in global_path],
            )

        # Odometry trail
        self._odom_xs.append(x)
        self._odom_ys.append(y)
        self._odom_line.set_data(self._odom_xs, self._odom_ys)

        # Remove old DWA candidate lines
        for ln in self._dwa_lines:
            ln.remove()
        self._dwa_lines.clear()

        if dwa_trajs:
            for traj_pts, valid in dwa_trajs:
                if len(traj_pts) < 2:
                    continue
                color = "lightgray" if valid else "salmon"
                (ln,) = self.ax.plot(
                    [p[0] for p in traj_pts],
                    [p[1] for p in traj_pts],
                    color=color, lw=0.6, alpha=0.45, zorder=2,
                )
                self._dwa_lines.append(ln)

        # Best DWA trajectory
        if best_traj and len(best_traj) >= 2:
            self._best_traj.set_data(
                [p[0] for p in best_traj],
                [p[1] for p in best_traj],
            )
        else:
            self._best_traj.set_data([], [])

        # Robot body + sensing circle
        self._robot_body.center = (x, y)
        self._sense_circle.center = (x, y)

        # Heading arrow
        arrow_len = self.robot_radius * 1.8
        dx = arrow_len * math.cos(theta)
        dy = arrow_len * math.sin(theta)
        self._arrow.remove()
        self._arrow = self.ax.annotate(
            "", xy=(x + dx, y + dy), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=2.0),
            zorder=7,
        )

        # Obstacle patches
        for p in self._obs_patches:
            p.remove()
        self._obs_patches.clear()

        for ox, oy, ir in obstacles:
            body = plt.Circle((ox, oy), 0.05, color="black", zorder=5)
            ring = plt.Circle(
                (ox, oy), ir,
                color="red", fill=False, linestyle="--", lw=1.5, zorder=5,
            )
            self.ax.add_patch(body)
            self.ax.add_patch(ring)
            self._obs_patches.extend([body, ring])

        self._flush()

    def close(self) -> None:
        plt.ioff()
        if not self._headless:
            plt.show(block=True)
        else:
            self.fig.savefig("/tmp/pa4_final_map.png", dpi=120)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        self.fig.canvas.draw_idle()
        plt.pause(0.001 if self._headless else 0.04)
