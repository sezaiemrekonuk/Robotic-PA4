# PA4 — Path Planning with A* and DWA Local Planner

## Quick-start

```bash
# 1. Build Docker image on the robot (or via dts)
dts devel build -f

# 2. Run on the Duckiebot
dts devel run -H ROBOT_NAME

# 3. To change mission parameters (A, B, obstacles):
#    Edit  packages/pa4_planner/config/params.yaml
#    — no code changes needed.
```

---

## Repository structure

```
packages/pa4_planner/
├── config/params.yaml          ← ALL tunable parameters (mission, robot, DWA, …)
├── launch/pa4_planner.launch   ← ROS launch file
├── setup.py                    ← catkin Python package install
└── src/
    ├── pa4_planner_node.py     ← main ROS node + state machine
    └── pa4_planner/
        ├── occupancy_grid.py   ← grid map with obstacle inflation
        ├── astar.py            ← 8-connected A* + path downsampling
        ├── dwa.py              ← Dynamic Window Approach local planner
        └── visualizer.py       ← real-time matplotlib display
```

---

## Module descriptions

### `occupancy_grid.py` — Grid map

Maintains a 2-D numpy array over the 1.5 × 1.5 m workspace (resolution chosen in params, default 0.05 m/cell → 30 × 30 grid).

| Cell value | Meaning |
|---|---|
| `FREE (0)` | Navigable |
| `INFLATED (1)` | Close to obstacle — DWA penalises but does not reject |
| `OCCUPIED (2)` | Inside inflated obstacle radius — A* and DWA reject |

Key methods:
- `add_obstacle(cx, cy, inflate_radius)` — marks cells; inflation = robot_radius + safety_margin.
- `world_to_cell / cell_to_world` — converts between metric world frame and integer grid indices.
- `collides_segment(x0,y0,x1,y1)` — used in the TURN phase to check if the direct path to the next waypoint is clear.

### `astar.py` — Global planner

Implements standard A* with an **8-connected neighbourhood** (straight cost = 1×res, diagonal = √2×res) and an **octile-distance heuristic** (admissible, consistent). Returns the shortest path as a list of world-frame (x, y) points.

`downsample_path()` reduces the dense cell-level path to human-scale waypoints by keeping only points where the direction changes more than `path.angle_threshold` radians or the accumulated spacing exceeds `path.min_spacing` metres. This controls how often the robot stops to turn and sense.

### `dwa.py` — Local planner

Implements the Dynamic Window Approach for a **unicycle (differential-drive)** robot.

**Algorithm per control step:**
1. Compute the dynamic window: feasible (v, ω) pairs reachable within one `dt` given the acceleration limits.
2. Sample an `v_samples × omega_samples` grid of (v, ω) pairs inside that window.
3. For each pair, simulate a trajectory of `horizon/dt` steps with the unicycle model `(x += v·cos(θ)·dt, etc.)`.
4. **Reject** any trajectory whose points enter an inflated obstacle.
5. **Score** valid trajectories:

   ```
   penalty = max(0, clearance_threshold − min_clearance_along_trajectory)

   score = −w_goal × dist(end, target_waypoint)
           − w_path × min_dist(end, global_path_points)
           − w_obs  × penalty     ← fires only when robot is close to obstacle
           + w_vel  × (v / v_max) ← rewards forward progress
   ```

   The obstacle term is **penalty-only** (active below `clearance_threshold`).
   A clearance-reward formulation would make staying still score higher than
   moving forward when far from obstacles — this design avoids that trap.

6. Execute the (v, ω) of the highest-scoring trajectory for one control step.

All weights and limits are exposed in `params.yaml`.

### `visualizer.py` — Real-time display

Single matplotlib figure, updated every control step, showing **simultaneously**:

| Layer | What |
|---|---|
| Grid background | Free (green) / Inflated (orange) / Occupied (red) occupancy |
| Blue line | Global A* path |
| Gray lines | All DWA candidate trajectories (salmon = rejected by obstacle) |
| Cyan line | Best DWA trajectory (chosen control) |
| Red circle + arrow | Robot body (radius to scale) + heading |
| Gold dashed circle | Sensing area |
| Black dot + red ring | Each obstacle centre + inflated boundary |
| Green trail | Odometry history (where the robot has actually been) |
| ▲ / ★ | Start and Goal markers |

Falls back to `Agg` backend (saves `/tmp/pa4_final_map.png`) if no display is available.

### `pa4_planner_node.py` — Main ROS node

Inherits from `DTROS` (Duckietown base). Loads config, wires up ROS subscribers/publishers, runs the state machine.

**Wheel odometry:** subscribes to left/right `WheelEncoderStamped` tick topics. Tick deltas are accumulated in callbacks (thread-safe lock) and consumed at each control step via `_process_odom()`. The midpoint integration formula `x += d·cos(θ + Δθ/2)` is used for accuracy.

**Wheel commands:** publishes `WheelsCmdStamped` with `vel_left / vel_right ∈ [−1, 1]`. Converts (v m/s, ω rad/s) using:
```
v_r = v + ω·L/2,   v_l = v − ω·L/2
cmd = v_wheel / speed_gain   (clipped to ±1)
```

#### State machine (per waypoint)

```
[INIT] Load config, add known obstacles, initial A* plan
  │
  └─► for each waypoint in plan:
        │
        ├─ TURN   Turn in-place toward waypoint.
        │          Check direction not blocked by known obstacles;
        │          if blocked → REPLAN before turning.
        │
        ├─ SENSE  Pause (sense_pause seconds). Scan forward arc
        │          for unknown obstacles within sensing_radius.
        │          ► If detected: add to map → REPLAN → restart.
        │          ► If clear: proceed.
        │
        └─ MOVE   DWA-guided motion to waypoint.
                   Also senses during motion (real-time unknown detection).
                   ► Reached: record pose in history, advance waypoint.
                   ► Timeout / DWA stuck 3 s: REPLAN.
                   ► Replan fails: BACKTRACK → pop history, move back,
                     replan. Repeat up to max_backtracks times.
                   ► max_backtracks exceeded: log "Dead end", stop, exit.
```

---

## Parameters reference (`config/params.yaml`)

| Section | Key | Effect |
|---|---|---|
| `map` | width, height, resolution | Grid dimensions and cell size |
| `start` / `goal` | [x, y] | Mission endpoints (instructor changes these) |
| `known_obstacles` | list of [x, y] | Added to map before A* |
| `unknown_obstacles` | list of [x, y] | **Bonus**: discovered by sensor at runtime |
| `robot` | radius, safety_margin | Obstacle inflation = radius + margin |
| `robot` | baseline, wheel_radius, ticks_per_rev | Odometry parameters (DB21 defaults) |
| `robot` | speed_gain | Tune per battery level |
| `robot` | slow_speed, turn_speed_cmd | Motion speed limits |
| `dwa` | v_samples, omega_samples | Trajectory sample density |
| `dwa` | weight_* | Cost function tuning |
| `sensing` | radius, fov_deg | Detection range and cone |
| `timing` | sense_pause | Dwell time at each waypoint |
| `path` | min_spacing, angle_threshold | Waypoint density after downsampling |
| `max_backtracks` | integer | Safety limit on backtrack depth |

---

## Bonus task — Unknown obstacle

Set `unknown_obstacles` in `params.yaml` with the obstacle position(s).  
These are **not** given to A* at startup. During `SENSE` (and also during `MOVE`), `_sense_obstacles()` checks whether any undiscovered obstacle is within `sensing.radius` metres and inside the forward FOV cone. When one is found:

1. Obstacle is added to the occupancy grid (inflated).
2. A* re-runs from current position — path automatically avoids it.
3. DWA continues with the updated obstacle list.

This simulates the real-robot pipeline: camera or ToF detects object → coordinates converted to world frame → same map-update + replan logic.

---

## How to change the mission (lab time)

Open `packages/pa4_planner/config/params.yaml` and modify:

```yaml
start: [xA, yA]
goal:  [xB, yB]

known_obstacles:
  - [xo, yo]

# For bonus (unknown obstacle):
unknown_obstacles:
  - [xo2, yo2]
```

Rebuild and rerun. No other files need to change.
