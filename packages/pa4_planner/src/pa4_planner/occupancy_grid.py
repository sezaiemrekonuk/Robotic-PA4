"""Occupancy grid: FREE / INFLATED / OCCUPIED with world-frame helpers."""
import math
from typing import List, Tuple
import numpy as np


class OccupancyGrid:
    FREE = 0
    INFLATED = 1
    OCCUPIED = 2

    def __init__(self, width_m: float, height_m: float, resolution_m: float):
        self.width_m = width_m
        self.height_m = height_m
        self.res = resolution_m
        self.cols = int(math.ceil(width_m / resolution_m))
        self.rows = int(math.ceil(height_m / resolution_m))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        # (cx, cy, inflated_radius) for every obstacle added
        self.obstacles: List[Tuple[float, float, float]] = []

    # Coordinate conversion

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int(x / self.res)
        row = int(y / self.res)
        return (
            max(0, min(self.rows - 1, row)),
            max(0, min(self.cols - 1, col)),
        )

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (col + 0.5) * self.res, (row + 0.5) * self.res

    # Obstacle management

    def add_obstacle(self, cx: float, cy: float, inflate_radius: float) -> None:
        """Mark cells within inflate_radius as OCCUPIED, outer ring as INFLATED."""
        # avoid duplicate near-identical obstacles
        for ox, oy, _ in self.obstacles:
            if math.hypot(cx - ox, cy - oy) < self.res:
                return

        self.obstacles.append((cx, cy, inflate_radius))
        r_cells = int(math.ceil(inflate_radius / self.res)) + 2
        row_c, col_c = self.world_to_cell(cx, cy)

        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                r, c = row_c + dr, col_c + dc
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    continue
                wx, wy = self.cell_to_world(r, c)
                dist = math.hypot(wx - cx, wy - cy)
                if dist <= inflate_radius:
                    self.grid[r, c] = self.OCCUPIED
                elif dist <= inflate_radius * 1.4 and self.grid[r, c] == self.FREE:
                    self.grid[r, c] = self.INFLATED

    # Queries

    def is_free_cell(self, row: int, col: int) -> bool:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.grid[row, col] == self.FREE

    def is_free_world(self, x: float, y: float) -> bool:
        return self.is_free_cell(*self.world_to_cell(x, y))

    def nearest_obstacle_dist(self, x: float, y: float) -> float:
        """Euclidean distance to nearest obstacle centre (inf if none)."""
        if not self.obstacles:
            return float("inf")
        return min(math.hypot(x - ox, y - oy) for ox, oy, _ in self.obstacles)

    def collides_segment(self, x0: float, y0: float,
                          x1: float, y1: float) -> bool:
        """True if the straight segment from (x0,y0) to (x1,y1) hits any obstacle."""
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(dist / (self.res * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            if not self.is_free_world(x, y):
                return True
        return False

    def hard_blocks_segment(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        ignore_radius_m: float = 0.0,
    ) -> bool:
        """
        Stricter than ``collides_segment``: only the OCCUPIED (true obstacle)
        cells block the segment. INFLATED cells (soft buffer) are not treated
        as a hard block — the local planner handles those via cost penalty.

        ``ignore_radius_m``: skip samples within this distance of the segment
        START point. Used so the robot can move out of its own inflation
        halo (e.g. after registering a close ToF hit) without falsely
        reporting "blocked".
        """
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-6:
            return False
        steps = max(2, int(dist / (self.res * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            seg = t * dist
            if seg < ignore_radius_m:
                continue
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            r, c = self.world_to_cell(x, y)
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return True
            if self.grid[r, c] == self.OCCUPIED:
                return True
        return False

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()
