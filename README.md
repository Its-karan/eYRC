# eYRC 2026–27 | Niti Vahan

A collection of implementations for the **Niti Vahan (NV)** theme of the e-Yantra Robotics Competition (eYRC) 2026–27. This repository contains work for Tasks **1A, 1B, 1C, 2A, 2B, and 2C**, covering Ackermann steering, path tracking, lane detection, arena navigation, sign and obstacle detection, and ArUco-based parking.

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

- `--out`: Output CSV file path.
- `--log-rate`: Number of CSV rows per second (default: 10).
- `--schedule`: Script lane changes using the `time:lane` format.

Refer to the eYRC theme book for submission instructions.

---

### Task 1C — Lane Detection

**Objective:** Detect the lane centre and determine whether the dashed white centre line is to the left or right of the vehicle.

**Files**

| File | Description |
|---|---|
| `lane_detection.py` | Boilerplate; implement `detect_lane()` |
| `public/` | 20 video clips, each 640 × 480 pixels, 20 FPS, and 200 frames |

The video clips are included in the repository; no additional download or extraction is required.

**Implementation**

Complete the following function:

```python
def detect_lane(frame):
    return {
        "center_x": <int>,
        "lane": "left" | "right" | "unknown"
    }
```

Return values:
- `center_x`: x-coordinate of the lane centre in pixels; use `-1` if the lane is not detected.
- `lane`:
  - `"left"` if the dashed white centre line is to the **right** of the vehicle.
  - `"right"` if the dashed white centre line is to the **left** of the vehicle.
  - `"unknown"` if the lane cannot be determined.

`detect_lane()` must only process the frame and return a result. Do not use `cv2.imshow()`, `cv2.waitKey()`, `cv2.imwrite()`, or `print()` inside this function. Visualisation is handled by `draw_overlay()` through `process_video()`.

**Check dependencies**

```bash
conda activate NV_3576
cd task1c
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
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

Press `q` to stop visual playback when using `--show`.

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

The returned control values are:
- `left_angle`, `right_angle`: Front-wheel steering angles in radians. The steering joints are positive to the left. The vehicle's steering limit is 45°.
- `speed`: Drive motor target in rad/s. A value of `0.0` holds the vehicle stationary.
- `stop_requested`: A level signal. When it is `True`, keep the vehicle stationary for as long as it remains `True`; do not implement your own stop timer.

Use the Ackermann geometry from Task 1A to split the steering angle between the two front wheels.

**Requirements**
- The `NV_3576` environment created for Task 0 should already contain the required dependencies.
- Check the environment with:

```bash
conda activate NV_3576
python -c "import cv2, numpy, coppeliasim_zmqremoteapi_client; print('ok')"
```

**Run**
1. Open CoppeliaSim and load `Task_2A.ttt`.
2. Press **Play** in CoppeliaSim.
3. Run the script:

```bash
conda activate NV_3576
python task2a.py
```

To reset the vehicle to its starting point between runs, press **Stop** and then **Play** in CoppeliaSim.

**Evaluation notes**
- Lane-centering accuracy is a major part of the evaluation. The vehicle is measured against a centreline surveyed from the road edges and dashed divider.
- Staying on the correct side of the divider alone is not sufficient; weaving within the lane reduces marks.
- The inner lane carries more marks than the outer lane.
- The evaluation tool records two laps. The score is calculated on the portal after the recording is uploaded; the local tool does not print a score.

**Important restriction:** The evaluator controls scene resets and simulation start/stop. Do not call `sim.startSimulation()`, `sim.stopSimulation()`, or `sim.setObjectPosition()` on the vehicle from your script.

---

### Task 2B — Sign and Obstacle Detection

**Objective:** Detect traffic signs and obstacles in the supplied images.

**Files**

| File or folder | Description |
|---|---|
| `sign_obstacle_detection.py` | Boilerplate; implement `detect_signs()` and `detect_obstacle()` |
| `public/` | Public images for development and testing |
| `reference_images/` | Parking and Hospital reference crops (`P.jpg`, `Hospital.png`) |

The reference images are intended for implementations that use image matching. They are not part of the submission ZIP; consult the official Submission Procedure.

**Implementation**

Complete these two functions:

```python
def detect_signs(image):
    return [{"label": <one of SIGN_LABELS>, "area": <int>}, ...]

def detect_obstacle(image):
    return {"present": <bool>, "area": <int>}
```

**Sign detection**
- Return one entry for each sign visible in the image.
- Return an empty list if no signs are detected.
- An image may contain more than one sign.
- Valid labels are defined by `SIGN_LABELS`:

```python
("School", "Speed40", "Parking", "Hospital")
```

**Obstacle detection**
- Return whether an obstacle is present and its area in pixels.
- Set `area` to `0` when `present` is `False`.

Both functions must only compute and return their results. Do not use `cv2.imshow()`, `cv2.waitKey()`, `cv2.imwrite()`, or `print()` inside either function.

**Check dependencies**

```bash
conda activate NV_3576
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

**Evaluate**

There is no standalone `main()` in this boilerplate. Run the official evaluation tool against the public images:

```bash
eyantra-autoeval evaluate --task 2b
```

---

### Task 2C — Parking with ArUco

**Objective:** Park the vehicle in one of two parking bays by using the ArUco marker associated with the target bay.

**Files**

| File | Description |
|---|---|
| `Task_2C.ttt` | CoppeliaSim scene containing the road, two parking bays, ArUco markers, and the vehicle |
| `task2c.py` | Boilerplate; implement `parking_manoeuvre()` |

**What the boilerplate provides**
- Camera reading and marker detection.
- The vehicle's pose relative to the target parking bay.
- Steering-angle splitting across the two front wheels using the Task 1A geometry.

**Implementation**

Complete `parking_manoeuvre(pose)`. The `pose` argument contains the vehicle's position and orientation relative to the target bay:

| Key | Description |
|---|---|
| `lateral` | Distance from the bay centre line in metres; positive means left of the line |
| `heading` | Heading relative to the bay's long axis in radians; positive means turned left |
| `forward` | Distance from the vehicle centre to the marker line in metres |
| `bearing` | Marker position in the camera view in radians; positive means left |

`pose` is `None` in frames where the marker is not detected.

The function must return:
1. A steering angle in radians (positive means left).
2. A speed command in rad/s.
3. A boolean indicating whether the vehicle is parked.

**Parking constraints**
- Bay width: 0.22 m.
- Vehicle width: 0.18 m.
- Approximate clearance: 20 mm on each side.
- Align both `lateral` and `heading` with zero; steering directly toward the marker creates a diagonal approach that will not fit within the bay.

**Run**
1. Open `Task_2C.ttt` in CoppeliaSim.
2. Keep the simulation **stopped**.
3. Run the script for the required bay:

```bash
conda activate NV_3576
cd task2c
python task2c.py --bay 0
```

When the script displays `Waiting for the simulation to start - press Play in CoppeliaSim.`, press **Play**.

Test both bays:

```bash
python task2c.py --bay 0
python task2c.py --bay 1
```

If `--bay` is omitted, the script uses `TARGET_ID` defined near the top of the file.

**Rules**
- Write code only inside the `ADD YOUR IMPLEMENTATION HERE` block.
- Do not move the vehicle using `sim.setObjectPosition()`.
- Do not read positions directly from the simulator; use the supplied `pose`.
- Do not use `print()`, plots, or `input()` inside `parking_manoeuvre()` when submitting.

---

## Prerequisites

The task instructions refer to the following software and Python packages:

- Python in the `NV_3576` Conda environment created for Task 0.
- CoppeliaSim for Tasks 1B, 2A, and 2C.
- OpenCV (`cv2`) and NumPy for the image-processing tasks.
- `coppeliasim_zmqremoteapi_client` for the Task 2A simulator connection.
- The `eyantra-autoeval` command-line tool for Task 2B evaluation.

The task notes indicate that the Task 0 environment should already contain the required dependencies. If a dependency is missing, consult the official eYRC setup instructions rather than installing unverified versions.

## Environment setup

Activate your team environment before running a task:

```bash
conda activate NV_3576
```

Replace `<Team-ID>` with your actual team identifier. Then move into the relevant task directory before running its script.

## Repository layout

The task instructions identify the following files and folders. Your actual repository may contain additional files:

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
│       ├── clip_01.mp4
│       └── ...
├── task2a/
│   ├── task2a.py
│   └── Task_2A.ttt
├── task2b/
│   ├── sign_obstacle_detection.py
│   ├── public/
│   └── reference_images/
└── task2c/
    ├── task2c.py
    └── Task_2C.ttt
```

This is an indicative layout based on the supplied task notes; adjust it to match the actual folders and filenames in your repository.

## Important implementation notes

- Implement only the functions specified for each task.
- Keep provided boilerplate, evaluation code, and simulator interfaces unchanged unless the official task instructions explicitly permit changes.
- Do not add visualisation, logging, or interactive input inside functions where the task rules prohibit it.
- Use the provided constants and pose/detection data instead of hardcoding values or reading restricted simulator state.
- Check the official theme book for the complete rules, marking criteria, and submission procedure.

## Submission

Task-specific submission requirements differ:

- **Task 1A:** Rename the completed file to `NV_Task1A.py`.
- **Tasks 1B, 1C, 2A, 2B, and 2C:** Follow the relevant Submission page or Submission Procedure in the official eYRC theme book.

Before submitting, verify that your implementation follows the task-specific restrictions and that the required files are named and packaged as specified by eYRC.

---

**Niti Vahan | eYRC 2026–27**
