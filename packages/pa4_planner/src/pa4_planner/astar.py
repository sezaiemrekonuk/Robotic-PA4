"""8-connected A* on OccupancyGrid with octile heuristic."""
import heapq
import math
from typing import List, Tuple, Optional
from .occupancy_grid import OccupancyGrid


def astar(
    grid: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
) -> Optional[List[Tuple[float, float]]]:
    """
    Returns a list of world (x, y) waypoints start→goal, or None if unreachable.
    Uses an 8-connected neighbourhood; diagonal cost = √2 × resolution.
    """
    start_cell = grid.world_to_cell(*start_world)
    goal_cell = grid.world_to_cell(*goal_world)

    # Relax goal if it sits inside an obstacle
    goal_cell = _nearest_free(grid, goal_cell) or goal_cell

    if not grid.is_free_cell(*start_cell):
        return None

    DIRS = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def octile(r: int, c: int) -> float:
        dr, dc = abs(r - goal_cell[0]), abs(c - goal_cell[1])
        return (dr + dc + (math.sqrt(2) - 2) * min(dr, dc)) * grid.res

    open_heap: List[Tuple[float, float, tuple]] = []
    heapq.heappush(open_heap, (octile(*start_cell), 0.0, start_cell))
    g_score: dict[tuple, float] = {start_cell: 0.0}
    came_from: dict[tuple, tuple] = {}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal_cell:
            return _reconstruct(grid, came_from, current, start_cell)

        if g > g_score.get(current, float("inf")):
            continue

        for dr, dc, step_cost in DIRS:
            nr, nc = current[0] + dr, current[1] + dc
            neighbour = (nr, nc)
            if not grid.is_free_cell(nr, nc):
                continue
            new_g = g + step_cost * grid.res
            if new_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = new_g
                came_from[neighbour] = current
                heapq.heappush(open_heap, (new_g + octile(nr, nc), new_g, neighbour))

    return None  # no path found


def _reconstruct(
    grid: OccupancyGrid,
    came_from: dict,
    current: tuple,
    start: tuple,
) -> List[Tuple[float, float]]:
    path = []
    while current in came_from:
        path.append(grid.cell_to_world(*current))
        current = came_from[current]
    path.append(grid.cell_to_world(*start))
    path.reverse()
    return path


def _nearest_free(
    grid: OccupancyGrid, cell: Tuple[int, int], max_r: int = 3
) -> Optional[Tuple[int, int]]:
    """BFS to find the nearest free cell within max_r cells."""
    if grid.is_free_cell(*cell):
        return cell
    from collections import deque
    queue = deque([cell])
    visited = {cell}
    while queue:
        r, c = queue.popleft()
        if abs(r - cell[0]) + abs(c - cell[1]) > max_r:
            break
        if grid.is_free_cell(r, c):
            return (r, c)
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nb = (r+dr, c+dc)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return None


def downsample_path(
    path: List[Tuple[float, float]],
    min_spacing: float,
    angle_threshold: float,
) -> List[Tuple[float, float]]:
    """
    Reduces A* path density.
    Keeps waypoints where direction changes > angle_threshold
    or distance since last kept point >= min_spacing.
    """
    if len(path) <= 2:
        return path

    result = [path[0]]
    dist_acc = 0.0

    for i in range(1, len(path) - 1):
        dx = path[i][0] - result[-1][0]
        dy = path[i][1] - result[-1][1]
        dist_acc += math.hypot(dx, dy)

        # Direction of segment i→i+1
        fwd_dx = path[i + 1][0] - path[i][0]
        fwd_dy = path[i + 1][1] - path[i][1]
        # Direction of segment (result[-1])→i
        bwd_dx = path[i][0] - result[-1][0]
        bwd_dy = path[i][1] - result[-1][1]

        angle = abs(math.atan2(fwd_dy, fwd_dx) - math.atan2(bwd_dy, bwd_dx))
        angle = min(angle, 2 * math.pi - angle)

        if dist_acc >= min_spacing or angle > angle_threshold:
            result.append(path[i])
            dist_acc = 0.0

    result.append(path[-1])
    return result
