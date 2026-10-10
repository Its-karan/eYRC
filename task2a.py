'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          e#YRC3576
# Author List:      Pankaj Amrate,Karan Singh,Krishna Sharma,Prasoon Dhakad
# Filename:         task2a.py
# Functions:        detect_lane, compute_control
# Global variables: none


####################### IMPORT MODULES #######################
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'

STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

# ---------------------------------------------------------------------------
# Vehicle constants – exact values from NV_Task1A.py / path_tracking.py
# ---------------------------------------------------------------------------
import math as _math

_WHEELBASE    = 0.120     # L metres (Task 1A / Task 1B)
_TRACK_WIDTH  = 0.110     # W metres
_WHEEL_OFFSET = 0.0275    # O metres (kingpin to wheel centre)

_MAX_STEER    = _math.radians(40.0)   # steering limit (radians)
_CRUISE_SPEED = 1.8                  # target velocity for motors (rad/s)


# ---------------------------------------------------------------------------
# State tracking across frames
# ---------------------------------------------------------------------------
class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.lane = None           # 'outer' or 'inner'
        self.lane_votes = {'outer': 0, 'inner': 0}
        self.frames_seen = 0
        self.prev_e_lat = 0.0
        self.last_delta = 0.0
        self.last_valid_cx_near = 320.0
        self.last_valid_cx_far = 320.0
        self.dropout_ctr = 0

_S = _State()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ackermann_wheel_angles(delta):
    """Split virtual front-centre steering angle into left and right wheel angles.
    Positive turns LEFT.
    """
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    l = _WHEELBASE
    w = _TRACK_WIDTH - 2.0 * _WHEEL_OFFSET
    s, c = _math.sin(delta), _math.cos(delta)
    left = _math.atan2(2.0 * l * s, 2.0 * l * c - w * s)
    right = _math.atan2(2.0 * l * s, 2.0 * l * c + w * s)
    return float(left), float(right)


# ---------------------------------------------------------------------------
# detect_lane
# ---------------------------------------------------------------------------
def detect_lane(frame):
    """
    Purpose:
    ---
    Work out, from a single camera frame, the lane centre line near the vehicle
    and ahead of the vehicle, so compute_control() can steer smoothly and
    accurately.
    """
    if frame is None or frame.size == 0:
        _S.dropout_ctr += 1
        return {
            'cx_near': _S.last_valid_cx_near,
            'cx_far': _S.last_valid_cx_far,
            'valid': False,
            'lane': _S.lane or 'outer'
        }

    _S.frames_seen += 1

    # 1. HSV color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Continuous yellow borders on inner and outer track edges
    mask_y = cv2.inRange(hsv, np.array([15, 80, 80], dtype=np.uint8),
                              np.array([40, 255, 255], dtype=np.uint8))
    # Dashed white divider separating the two lanes
    mask_w = cv2.inRange(hsv, np.array([0, 0, 180], dtype=np.uint8),
                              np.array([180, 40, 255], dtype=np.uint8))

    # Mask out irrelevant regions (sky, horizon, bumper)
    mask_y[:180, :] = 0; mask_y[450:, :] = 0
    mask_w[:180, :] = 0; mask_w[450:, :] = 0

    # 2. Lane determination (initial voting)
    # In outer lane: yellow border is on left (lower col), white divider is on right
    # In inner lane: white divider is on left, yellow border is on right
    if _S.lane is None:
        y_cols = np.where(mask_y[320:440, :] > 0)[1]
        w_cols = np.where(mask_w[320:440, :] > 0)[1]
        y_m = np.median(y_cols) if len(y_cols) > 30 else 320.0
        w_m = np.median(w_cols) if len(w_cols) > 30 else 320.0
        if y_m < w_m:
            _S.lane_votes['outer'] += 1
        else:
            _S.lane_votes['inner'] += 1

        if _S.frames_seen >= 5:
            _S.lane = 'outer' if _S.lane_votes['outer'] >= _S.lane_votes['inner'] else 'inner'

    active_lane = _S.lane if _S.lane is not None else ('outer' if _S.lane_votes['outer'] >= _S.lane_votes['inner'] else 'inner')

    # 3. Multi-band lane centre extraction
    def extract_band_cx(r_top, r_bot, r_mid):

        # Estimate marking positions row by row.
        def robust_x(mask):
            row_centres = []

            for row in range(r_top, r_bot):
                cols = np.where(mask[row, :] > 0)[0]

                if len(cols) >= 3:
                    row_centres.append(float(np.median(cols)))

            if len(row_centres) < 4:
                return None

            return float(np.median(row_centres))

        y_pos = robust_x(mask_y)
        w_pos = robust_x(mask_w)

        # Approximate half lane width.
        w_half = 140.0 + 0.6 * (r_mid - 280.0)

        if y_pos is not None and w_pos is not None:
            return (y_pos + w_pos) / 2.0

        elif active_lane == 'outer':

            if y_pos is not None:
                return y_pos + w_half

            if w_pos is not None:
                return w_pos - w_half

        else:

            if w_pos is not None:
                return w_pos + w_half

            if y_pos is not None:
                return y_pos - w_half

        return None
    # Near band (row 360-400, midpoint 380) and Far band (row 260-300, midpoint 280)
    cx_near = extract_band_cx(360, 400, 380.0)
    cx_far  = extract_band_cx(300, 340, 320.0)

    valid = True
    if cx_near is None and cx_far is None:
        _S.dropout_ctr += 1
        cx_near = _S.last_valid_cx_near
        cx_far  = _S.last_valid_cx_far
        valid = False
    else:
        _S.dropout_ctr = 0
        if cx_near is None: cx_near = cx_far
        if cx_far is None:  cx_far = cx_near
        _S.last_valid_cx_near = cx_near
        _S.last_valid_cx_far  = cx_far

    return {
        'cx_near': float(cx_near),
        'cx_far':  float(cx_far),
        'valid':   valid,
        'lane':    active_lane
    }


# ---------------------------------------------------------------------------
# compute_control
# ---------------------------------------------------------------------------
def compute_control(detection, stop_requested):
    """
    Stable lane-following controller with:
    - Lateral error correction
    - Lookahead curvature compensation
    - Reduced derivative sensitivity
    - Smoothed steering
    - Adaptive motor speed
    """

    # 1. Stop immediately when requested by the evaluation system.
    if stop_requested:
        left_angle, right_angle = _ackermann_wheel_angles(
            _S.last_delta
        )
        return float(left_angle), float(right_angle), 0.0

    # 2. Handle temporary lane-detection failures.
    if detection is None or not detection.get('valid', False):

        if _S.dropout_ctr > 8:
            # Stop if the lane has been lost for too long.
            left_angle, right_angle = _ackermann_wheel_angles(
                _S.last_delta
            )
            return float(left_angle), float(right_angle), 0.0

        cx_near = _S.last_valid_cx_near
        cx_far = _S.last_valid_cx_far

    else:
        cx_near = detection['cx_near']
        cx_far = detection['cx_far']

    # 3. Prevent invalid image coordinates.
    cx_near = max(0.0, min(639.0, float(cx_near)))
    cx_far = max(0.0, min(639.0, float(cx_far)))

    # 4. Calculate lateral error.
    # Positive error means the lane centre is to the LEFT
    # of the image centre, so the vehicle must steer LEFT.
    e_lat = (320.0 - cx_near) / 160.0

    # 5. Estimate the road direction using lookahead.
    # A positive angle means the lane centre moves RIGHT
    # in the image as we look farther ahead.
    dx = cx_far - cx_near

    theta_curve = _math.atan2(dx, 120.0)

    # 6. Calculate the change in lateral error.
    d_e = e_lat - _S.prev_e_lat
    _S.prev_e_lat = e_lat

    # 7. Steering controller.
    k_p = 1.9
    k_curve = 0.85
    k_d = 0.08

    raw_delta = (
        k_p * e_lat
        - k_curve * theta_curve
        + k_d * d_e
    )

    # 8. Limit the steering angle.
    raw_delta = max(
        -_MAX_STEER,
        min(_MAX_STEER, raw_delta)
    )

    # 9. Smooth steering to reduce sudden oscillations.
    delta = (
        0.35 * raw_delta
        + 0.65 * _S.last_delta
    )

    _S.last_delta = delta

    # 10. Convert virtual steering to Ackermann wheel angles.
    left_angle, right_angle = _ackermann_wheel_angles(delta)

    # 11. Adapt speed to steering demand.
    steer_ratio = abs(delta) / _MAX_STEER

    curve_ratio = min(
        1.0,
        abs(theta_curve) / 0.5
    )

    speed_factor = (
        1.0
        - 0.45 * steer_ratio
        - 0.25 * curve_ratio
    )

    speed = _CRUISE_SPEED * speed_factor

    # Keep a reasonable minimum speed during normal driving.
    speed = max(1.0, speed)

    # Do not apply a minimum speed if the stop signal is active;
    # that case has already returned above.

    return (
        float(left_angle),
        float(right_angle),
        float(speed)
    )

##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())