# Task 1A — Ackermann Steering

Fill in `ackermann_wheel_angles(delta)` in `ackermann_steering.py`.

```sh
conda activate NV_<3576>
cd ~/eYRC_26-27_Niti-Vahan/task1a
python ackermann_steering.py
```

No simulator needed — this subtask is pure geometry. Running the file executes
the test block at the bottom, which sweeps `delta` from -0.35 to 0.35 rad and
prints the pair of wheel angles your function returns for each.

| Constant | Value | Meaning |
|---|---:|---|
| `WHEELBASE` | 0.120 m | front axle to rear axle |
| `TRACK_WIDTH` | 0.110 m | left wheel centre to right wheel centre |
| `WHEEL_OFFSET` | 0.0275 m | kingpin axis to wheel centre |

Read these from the constants — do not hardcode the numbers.

**To submit:** rename your file to `NV_Task1A.py` and upload it. See the
Submission page in the theme book.

# Task 1B — Path Tracking

Fill in `ackermann_wheel_angles()` and `compute_steering()` in `path_tracking.py`.

## Setup

1. Launch **CoppeliaSim**.
2. **File → Open scene…** and pick `Task_1B.ttt` from this folder.
3. Leave the simulation **stopped** — do not press the ▶️ Play button. The
   script starts and stops it for you.
4. In a terminal:

```sh
conda activate NV_<Team-ID>
cd ~/eYRC_26-27_Niti-Vahan/task1b
python path_tracking.py
```

## The scene

| Object in the hierarchy | What it is |
|---|---|
| `/Floor` | the road, 5 m long and 1 m wide |
| `/Niti_Vahan` | the vehicle body |
| `/Niti_Vahan/steeringLeft`, `steeringRight` | front steering joints — your wheel angles go here |
| `/Niti_Vahan/motorLeft`, `motorRight` | driven front wheels, held at a constant speed |
| `/Niti_Vahan/freeAxisLeft`, `freeAxisRight` | rear wheels, free-spinning |

The vehicle drives along world **-x**, so its left-hand side faces **-y**.

| Lane | Lateral position |
|:---:|---|
| `L` | y = 0.00 m |
| `R` | y = 0.20 m |

## Driving it

While the run is going, type `L` or `R` and press Enter to change lane, `q` to
stop. Every run lasts 120 simulated seconds and writes `trajectory.csv`.

```sh
python path_tracking.py --out run1.csv     # write the CSV elsewhere
python path_tracking.py --log-rate 20      # rows per second (default 10)
python path_tracking.py --schedule "15:R,60:L,95:R"  # scripted lane changes
```

**To submit:** see the Submission page in the theme book.

# Task 1C - Lane Detection Boilerplate

Niti Vahan (NV), eYRC 2026-27.

## Files

| File | What it is |
|---|---|
| `lane_detection.py` | The boilerplate. Fill in `detect_lane()`; leave everything else alone. |
| `public/` | The 20 video clips - 640 x 480, 20 fps, 200 frames each. |

## Setup

Nothing to download or unzip: cloning this repository gives you the clips.

```text
task1c/
├── lane_detection.py
└── public/
    ├── clip_01.mp4
    └── ...          (clip_20.mp4)
```

No simulator needed. Check your environment has what you need:

```sh
conda activate NV_<3576>
cd ~/eYRC_26-27_Niti-Vahan/task1c
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## Usage

```sh
# one clip
python lane_detection.py public/clip_01.mp4

# every clip in the folder (press q to stop) - a folder works on Windows too
python lane_detection.py public --show

# write the per-frame results to a file
python lane_detection.py public --out results.json
```

## What you implement

Exactly one function:

```python
def detect_lane(frame):
    return {"center_x": <int>, "lane": "left" | "right" | "unknown"}
```

- `center_x` - x-pixel of the lane centre in this frame, `-1` if the lane was not found.
- `lane` - `"left"` if the dashed white centre line is to the **right** of the vehicle,
  `"right"` if it is to the **left**, `"unknown"` if you cannot tell.

`detect_lane()` must only compute and return. No `cv2.imshow()`, `cv2.waitKey()`,
`cv2.imwrite()` or `print()` inside it - visualisation goes in `draw_overlay()`,
which is called from `process_video()`.

**To submit:** see the Submission page in the theme book.

# Task 2A - Arena Navigation Boilerplate

Niti Vahan (NV), eYRC 2026-27.

## Files

| File | What it is |
|---|---|
| `task2a.py` | The boilerplate. Fill in `detect_lane()` and `compute_control()`; leave everything else alone. |
| `Task_2A.ttt` | The CoppeliaSim scene. Open it before you run anything. |

## Requirements

Your `NV_<Team-ID>` environment from Task 0 already has what you need - nothing
new to install since Task 1:

```sh
conda activate NV_<Team-ID>
python -c "import cv2, numpy, coppeliasim_zmqremoteapi_client; print('ok')"
```

## Usage

1. Launch CoppeliaSim and open `Task_2A.ttt`.
2. Press **▶ Play**.
3. In a terminal:

```sh
conda activate NV_<Team-ID>
python task2a.py
```

Press **⏹ Stop** then **▶ Play** to put the vehicle back at its starting point
between runs.

## What you implement

Two functions, and nothing else:

```python
def detect_lane(frame):
    # frame is a corrected BGR image from the vehicle's camera
    return <anything compute_control() can use>

def compute_control(detection, stop_requested):
    return left_angle, right_angle, speed
```

- `left_angle`, `right_angle` - the two **front-wheel** angles, in **radians**.
  The scene's steering joints are **positive to the left**. Nothing is clamped
  for you; the vehicle's own limit is 45°. Use your Task 1A geometry to split
  one steering angle into these two.
- `speed` - drive motor target, in rad/s. `0.0` holds the vehicle still.
- `stop_requested` - `True` while the evaluation is holding you at a stop point.
  It is a **level**, not a pulse: stay stationary for as long as it is `True`.
  Do not time the stop yourself.

Everything below the `DO NOT EDIT BELOW THIS LINE` marker is provided: the
connection, the handle lookup, the camera read, the four joint writes and the
loop. None of it is a decision you are marked on - it is there so every team
reads the same picture and drives the same vehicle.

## Where the marks are

Most of Task 2A is **how closely you hold the middle of your lane**, measured
against a centreline surveyed from the painted road edges and the dashed
divider. Being on the correct side of the divider is not enough - weaving
across your own lane loses marks. The inner lane is worth more than the outer
one.

The evaluation tool on your machine **records** the two laps; the marks are
worked out on the portal when you upload the recording. So the tool will not
print a score - watch the simulator instead, and submit to see the marks.

## What your script must never do

The evaluation resets the scene, places the vehicle in the lane under test, and
starts and stops the simulation. Your script only reads the camera and drives
the joints. Calling `sim.startSimulation()`, `sim.stopSimulation()` or
`sim.setObjectPosition()` on the vehicle fights it for control of the run - the
evaluator reports these when it finds them.

Full instructions are in the theme book.

# Task 2B - Sign & Obstacle Detection Boilerplate

Niti Vahan (NV), eYRC 2026-27.

## Files

| File | What it is |
|---|---|
| `sign_obstacle_detection.py` | The boilerplate. Fill in `detect_signs()` and `detect_obstacle()`; leave everything else alone. |
| `public/` | The public set of images to develop against. |
| `reference_images/` | Parking/Hospital reference crops (`P.jpg`, `Hospital.png`), shipped for implementations that want to match against them. Not part of the submission zip - see Submission Procedure. |

## Requirements

Your `NV_<Team-ID>` environment from Task 0 already has what you need:

```sh
conda activate NV_<Team-ID>
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

## What you implement

Two functions:

```python
def detect_signs(image):
    return [{"label": <one of SIGN_LABELS>, "area": <int>}, ...]

def detect_obstacle(image):
    return {"present": <bool>, "area": <int>}
```

- `detect_signs()` - one entry per sign visible in the image; an empty list if there
  are none. An image can hold more than one sign at once.
  `SIGN_LABELS` is `("School", "Speed40", "Parking", "Hospital")`.
- `detect_obstacle()` - whether the obstacle is visible, and its area in pixels if so
  (`0` when `present` is `False`).

Both functions must only compute and return. No `cv2.imshow()`, `cv2.waitKey()`,
`cv2.imwrite()` or `print()` inside them - reading images and reporting results is
the evaluator's job, not this file's. There is no `main()` here to run directly;
use the evaluation tool (`eyantra-autoeval evaluate --task 2b`) to run your code
against the public images.

Full instructions are in the theme book.
