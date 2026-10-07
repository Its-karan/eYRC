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
