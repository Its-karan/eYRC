# eYRC 2026–27 | Niti Vahan

A collection of implementations for the **Niti Vahan (NV)** theme of the e-Yantra Robotics Competition (eYRC) 2026–27. This repository contains work for Tasks **1A, 1B, 1C, 2A, 2B, 2C, and Task 3**, covering Ackermann steering, path tracking, lane detection, arena navigation, sign and obstacle detection, ArUco-based parking, and the final integrated autonomous mission.

> **Note:** This README consolidates the task-level instructions provided with the project. Refer to the official eYRC theme book and submission procedure for complete evaluation and submission requirements.

---

## Table of contents

- [Project tasks](#project-tasks)
  - [Task 1A — Ackermann Steering](#task-1a--ackermann-steering)
  - [Task 1B — Path Tracking](#task-1b--path-tracking)
  - [Task 1C — Lane Detection](#task-1c--lane-detection)
  - [Task 2A — Arena Navigation](#task-2a--arena-navigation)
  - [Task 2B — Sign and Obstacle Detection](#task-2b--sign-and-obstacle-detection)
  - [Task 2C — Parking with ArUco](#task-2c--parking-with-aruco)
  - [Task 3 — Full Autonomous Mission](#task-3--full-autonomous-mission)
- [Prerequisites](#prerequisites)
- [Environment setup](#environment-setup)
- [Repository layout](#repository-layout)
- [Important implementation notes](#important-implementation-notes)
- [Submission](#submission)

---

## Project tasks

### Task 1A — Ackermann Steering

**Objective:** Implement the Ackermann steering geometry that converts a steering input into the corresponding left and right front-wheel angles.

**Implementation**
- Complete `ackermann_wheel_angles(delta)` in `ackermann_steering.py`.
- Read the geometry values from the provided constants rather than hardcoding them.

| Constant | Value | Description |
|---|---:|---|
| `WHEELBASE` | 0.120 m | Distance from the front axle to the rear axle |
| `TRACK_WIDTH` | 0.110 m | Distance between the left and right wheel centres |
| `WHEEL_OFFSET` | 0.0275 m | Distance from the kingpin axis to the wheel centre |

**Run**

```bash
conda activate NV_3576
cd task1a
python ackermann_steering.py
```

No simulator is required. The test block sweeps `delta` from `-0.35` to `0.35` radians and prints the resulting wheel-angle pairs.

**Submission file:** Rename the completed file to `NV_Task1A.py`. Follow the Submission page in the eYRC theme book.

---

### Task 1B — Path Tracking

**Objective:** Implement Ackermann wheel-angle calculation and path-tracking steering control for the simulated vehicle.

**Implementation**
- Complete `ackermann_wheel_angles()` and `compute_steering()` in `path_tracking.py`.

**Scene setup**
1. Open CoppeliaSim.
2. Load `Task_1B.ttt` from the Task 1B folder.
3. Keep the simulation **stopped**. The script starts and stops it.

**Run**

```bash
conda activate NV_3576
cd task1b
python path_tracking.py
```

**Scene reference**

| Scene object | Description |
|---|---|
| `/Floor` | Road, 5 m long and 1 m wide |
| `/Niti_Vahan` | Vehicle body |
| `/Niti_Vahan/steeringLeft`, `/Niti_Vahan/steeringRight` | Front steering joints |
| `/Niti_Vahan/motorLeft`, `/Niti_Vahan/motorRight` | Driven front wheels, held at constant speed |
| `/Niti_Vahan/freeAxisLeft`, `/Niti_Vahan/freeAxisRight` | Free-spinning rear wheels |

The vehicle travels along world **−x**, with its left-hand side facing **−y**.

| Lane | Lateral position |
|:---:|---:|
| `L` | y = 0.00 m |
| `R` | y = 0.20 m |

**Controls**
- Enter `L` or `R` while the script is running to change lanes.
- Enter `q` to stop.
- Each run lasts 120 simulated seconds and writes `trajectory.csv` by default.

**Optional command-line arguments**

```bash
python path_tracking.py --out run1.csv
python path_tracking.py --log-rate 20
python path_tracking.py --schedule "15:R,60:L,95:R"
```

---

### Task 1C — Lane Detection

**Objective:** Detect the lane centre and determine whether the dashed white centre line is to the left or right of the vehicle.

**Files**

| File | Description |
|---|---|
| `lane_detection.py` | Boilerplate; implement `detect_lane()` |
| `public/` | 20 video clips, each 640 × 480 pixels, 20 FPS, and 200 frames |

**Implementation**

Complete the following function:

```python
def detect_lane(frame):
    return {
        "center_x": <int>,
        "lane": "left" | "right" | "unknown"
    }
```

**Run**

```bash
# Process one video
python lane_detection.py public/clip_01.mp4

# Process all clips in the folder
python lane_detection.py public --show

# Save per-frame results
python lane_detection.py public --out results.json
```

---

### Task 2A — Arena Navigation

**Objective:** Detect the lane and compute steering and speed commands to keep the vehicle aligned within the lane during arena navigation.

**Files**
- `task2a.py` — boilerplate containing the functions to implement.
- `Task_2A.ttt` — CoppeliaSim scene.

**Implementation**

Complete only these two functions:

```python
def detect_lane(frame):
    # frame is a corrected BGR image from the vehicle's camera
    return detection

def compute_control(detection, stop_requested):
    return left_angle, right_angle, speed
```

**Run**
1. Open CoppeliaSim and load `Task_2A.ttt`.
2. Press **Play** in CoppeliaSim.
3. Run the script:

```bash
conda activate NV_3576
python task2a.py
```

---

### Task 2B — Sign and Obstacle Detection

**Objective:** Detect traffic signs and obstacles in the supplied images.

**Files**

| File or folder | Description |
|---|---|
| `sign_obstacle_detection.py` | Boilerplate; implement `detect_signs()` and `detect_obstacle()` |
| `public/` | Public images for development and testing |

**Implementation**

Complete these two functions:

```python
def detect_signs(image):
    return [{"label": <one of SIGN_LABELS>, "area": <int>}, ...]

def detect_obstacle(image):
    return {"present": <bool>, "area": <int>}
```

**Evaluate**

```bash
eyantra-autoeval evaluate --task 2b
```

---

### Task 2C — Parking with ArUco

**Objective:** Park the vehicle in one of two parking bays by using the ArUco marker associated with the target bay.

**Files**

| File | Description |
|---|---|
| `Task_2C.ttt` | CoppeliaSim scene |
| `task2c.py` | Boilerplate; implement `parking_manoeuvre()` |

**Run**
1. Open `Task_2C.ttt` in CoppeliaSim.
2. Keep the simulation **stopped**.
3. Run the script for the required bay:

```bash
conda activate NV_3576
cd task2c
python task2c.py --bay 0
```

---

### Task 3 — Full Autonomous Mission

**Objective:** Integrate lane tracking, traffic sign perception, obstacle avoidance, and ArUco-based parking to complete a full autonomous lap in three varying difficulty arenas.

**Files**
- `task3.py` — The unified mission controller.
- `Task_3_Easy.ttt`, `Task_3_Medium.ttt`, `Task_3_Hard.ttt` — CoppeliaSim arenas.

**Implementation Highlights**
- **State Machine Architecture:** Manages seamless transitions between cruising, obstacle avoidance, traffic halts, and bay docking (`LANE_FOLLOWING`, `HALTING_HOSPITAL`, `LANE_CHANGING`, `APPROACH_PARKING`, `PARKING_MANEUVER`, `PARKED`).
- **Strict Traffic Rule Compliance:** Implements a mandatory 5.0-second halt at the Hospital sign, speed limit regulation (≤ 1.2 rad/s) between Speed 40 signs, and safe passage for School signs.
- **Dynamic Obstacle Avoidance:** Utilizes morphological closing and dynamic ROI targeting to detect 3D red car obstacles on the roadway. Executes a closed-loop cosine S-curve lateral shift to the adjacent lane, permanently retaining the new lane post-dodge to satisfy strict competition rules.
- **Precision Bay Parking:** Triggers a bay search sequence upon detecting a Parking sign. Acquires the target ArUco marker (ID 1) and executes a 3-zone S-curve docking controller for dead-center parking without side wall collisions.

**Run**
1. Open CoppeliaSim and load the desired Task 3 arena (e.g., `Task_3_Hard.ttt`).
2. Run the script (it will automatically start the simulation):

```bash
conda activate NV_3576
cd task3
python task3.py
```

---

## Prerequisites

The task instructions refer to the following software and Python packages:

- Python in the `NV_3576` Conda environment created for Task 0.
- CoppeliaSim for Tasks 1B, 2A, 2C, and 3.
- OpenCV (`cv2`) and NumPy for the image-processing tasks.
- `coppeliasim_zmqremoteapi_client` for the simulator connection.
- The `eyantra-autoeval` command-line tool for evaluation.

## Environment setup

Activate your team environment before running a task:

```bash
conda activate NV_3576
```

## Repository layout

```text
.
├── task1a/
│   └── ackermann_steering.py
├── task1b/
│   ├── path_tracking.py
│   └── Task_1B.ttt
├── task1c/
│   ├── lane_detection.py
│   └── public/
├── task2a/
│   ├── task2a.py
│   └── Task_2A.ttt
├── task2b/
│   ├── sign_obstacle_detection.py
│   ├── public/
│   └── reference_images/
├── task2c/
│   ├── task2c.py
│   └── Task_2C.ttt
└── task3/
    ├── task3.py
    ├── Task_3_Easy.ttt
    ├── Task_3_Medium.ttt
    └── Task_3_Hard.ttt
```

## Important implementation notes

- Implement only the functions specified for each task.
- Keep provided boilerplate, evaluation code, and simulator interfaces unchanged unless the official task instructions explicitly permit changes.
- Use the provided constants and pose/detection data instead of hardcoding values or reading restricted simulator state.
- Check the official theme book for the complete rules, marking criteria, and submission procedure.

## Submission

Task-specific submission requirements differ:

- **Task 1A:** Rename the completed file to `NV_Task1A.py`.
- **Tasks 1B, 1C, 2A, 2B, 2C, and Task 3:** Follow the relevant Submission page or Submission Procedure in the official eYRC theme book. Ensure that your final `task3.py` script correctly imports and integrates all required sub-modules.
