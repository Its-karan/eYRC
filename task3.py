"""
===============================================================================
eYRC Niti Vahan Competition 2026-27 — Task 3
Autonomous Navigation, Traffic Rule Compliance, Obstacle Avoidance & Bay Parking
Team ID : eYRC#3576
Authors : Pankaj Amrate, Karan Singh, Prasoon Dhakad, Krishna Sharma
Filename: task3.py
Arenas  : Task_3_Easy.ttt  |  Task_3_Medium.ttt  |  Task_3_Hard.ttt
===============================================================================
"""

import math
import sys
import time
from enum import Enum

import cv2
import numpy as np

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    print("[ERROR] coppeliasim_zmqremoteapi_client is not installed.")
    sys.exit(1)

# =============================================================================
# §1  VEHICLE GEOMETRY
# =============================================================================

WHEELBASE             = 0.120     # L  metres
TRACK_WIDTH           = 0.110     # W  metres
WHEEL_OFFSET          = 0.0275    # O  metres  (kingpin to wheel centre)
MAX_STEER             = math.radians(40.0)
MAX_LANE_CHANGE_STEER = math.radians(28.0)  # Smooth shift without losing turn authority
MAX_STEER_BAY         = 0.15                # Maximum steering inside bay walls (rad)

# =============================================================================
# §2  SPEED CONSTANTS  (rad/s)
# =============================================================================

BASE_CRUISE_SPEED      = 1.60   # Normal cruise speed (<= 3.0 rad/s limit)
SPEED_ZONE_SPEED       = 1.15   # Inside Speed-40 zone (<= 1.2 rad/s limit)
LANE_CHANGE_SPEED      = 1.15   # Controlled speed during S-curve lane change
PARKING_APPROACH_SPEED = 0.85   # Controlled approach speed entering parking straight
PARKING_DOCK_SPEED     = 0.45   # Creep inside bay

# =============================================================================
# §3  TIMING & COUNTING CONSTANTS
# =============================================================================

HOSPITAL_HALT_DURATION      = 5.0    # Mandatory stop (sim seconds)
HOSPITAL_SUPPRESS_WINDOW    = 35.0   # Sim seconds to suppress Hospital sign after halt
LANE_CHANGE_DURATION        = 7.5    # Sim seconds for smooth S-curve lane shift
OBSTACLE_COOLDOWN_TIME      = 14.0   # Sim seconds cooldown after lane change starts
OBSTACLE_REARM_FRAMES       = 8      # Obstacle-free frames in active lane to re-arm
OBSTACLE_CONFIRM_FRAMES     = 2      # Consecutive frames to confirm obstacle in lane
SPEED_SIGN_MIN_AREA         = 115    # px^2 min area to detect Speed40 from either lane
SPEED_SIGN_REARM_FRAMES     = 15     # Frames of absence to re-arm Speed40 sign
SPEED_SIGN_COOLDOWN_SECONDS = 15.0   # Sim seconds between 1st and 2nd Speed40 sign
PARKING_SEARCH_DELAY        = 65.0   # Sim seconds after 2nd Speed40 sign before bay search
ARUCO_CONFIRM_FRAMES        = 2      # Consecutive valid-pose frames to lock onto Bay ID_1

# =============================================================================
# §4  CAMERA & ARUCO CONSTANTS
# =============================================================================

ARUCO_DICT        = cv2.aruco.DICT_4X4_50
TARGET_BAY_ID     = 1
IMG_W, IMG_H      = 640, 480
FOCAL_PX          = (IMG_W / 2.0) / math.tan(math.radians(95.0) / 2.0)
CAMERA_MATRIX     = np.array([[FOCAL_PX, 0.0, IMG_W / 2.0],
                              [0.0, FOCAL_PX, IMG_H / 2.0],
                              [0.0, 0.0,      1.0        ]], dtype=np.float32)
CAM_PITCH         = math.radians(30.0)
CAM_ABOVE_MK      = 0.176 - 0.110
CAM_AHEAD         = 0.047
MARKER_SIDE       = 0.096
_HM               = MARKER_SIDE / 2.0
MARKER_CORNERS_3D = np.array([[-_HM,  _HM, 0.0],
                              [ _HM,  _HM, 0.0],
                              [ _HM, -_HM, 0.0],
                              [-_HM, -_HM, 0.0]], dtype=np.float32)
YAW_BLEND          = 0.12
YAW_GATE           = math.radians(15.0)
AMBIGUITY_RATIO    = 1.5
BAY_MOUTH_FORWARD  = 0.37
BAY_MIDDLE_FORWARD = 0.19

# =============================================================================
# §5  SCENE OBJECT HANDLE CANDIDATES
# =============================================================================

VISION_SENSOR_CANDIDATES = [
    '/Niti_Vahan/Camera_joint/Cuboid/visionSensor',
    '/vision_sensor',
    '/Niti_Vahan/vision_sensor',
]
STEER_LEFT_CANDIDATES  = ['/steeringLeft',  '/Niti_Vahan/steeringLeft']
STEER_RIGHT_CANDIDATES = ['/steeringRight', '/Niti_Vahan/steeringRight']
MOTOR_LEFT_CANDIDATES  = [
    '/motorLeft',
    '/Niti_Vahan/steeringLeft/Cylinder/motorLeft',
    '/Niti_Vahan/motorLeft',
]
MOTOR_RIGHT_CANDIDATES = [
    '/motorRight',
    '/Niti_Vahan/steeringRight/Cylinder/motorRight',
    '/Niti_Vahan/motorRight',
]

# =============================================================================
# §6  STATE ENUM
# =============================================================================

class RobotState(Enum):
    LANE_FOLLOWING   = "LANE_FOLLOWING"
    HALTING_HOSPITAL = "HALTING_HOSPITAL"
    LANE_CHANGING    = "LANE_CHANGING"
    PARKING_MANEUVER = "PARKING_MANEUVER"
    PARKED           = "PARKED"

# =============================================================================
# §7  DRIVE & ACTUATION HELPERS
# =============================================================================

def ackermann_wheel_angles(delta: float):
    """
    Virtual centre steering angle -> (left, right) wheel angles.
    Positive delta turns LEFT; Negative delta turns RIGHT.
    """
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    l, w = WHEELBASE, TRACK_WIDTH - 2.0 * WHEEL_OFFSET
    s, c = math.sin(delta), math.cos(delta)
    return (float(math.atan2(2.0 * l * s, 2.0 * l * c - w * s)),
            float(math.atan2(2.0 * l * s, 2.0 * l * c + w * s)))


def drive(sim, handles: dict, steering_angle: float, speed: float):
    la, ra = ackermann_wheel_angles(steering_angle)
    sim.setJointTargetPosition(handles['steer_left'],  la)
    sim.setJointTargetPosition(handles['steer_right'], ra)
    sim.setJointTargetVelocity(handles['motor_left'],  float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def stop_vehicle(sim, handles: dict):
    try:
        sim.setJointTargetPosition(handles['steer_left'],  0.0)
        sim.setJointTargetPosition(handles['steer_right'], 0.0)
        sim.setJointTargetVelocity(handles['motor_left'],  0.0)
        sim.setJointTargetVelocity(handles['motor_right'], 0.0)
    except Exception:
        pass


def read_frame(sim, vision_sensor_handle: int):
    """Return (bgr, hsv) or (None, None) on failure."""
    try:
        sim.handleVisionSensor(vision_sensor_handle)
        img_bytes, resolution = sim.getVisionSensorImg(vision_sensor_handle, 0)
    except Exception:
        return None, None
    if not img_bytes or len(img_bytes) == 0:
        return None, None
    frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
        (resolution[1], resolution[0], 3))
    frame = cv2.flip(frame, 0)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return bgr, hsv

# =============================================================================
# §8  COLOUR HELPERS
# =============================================================================

def _isolate_red(hsv) -> np.ndarray:
    """Red isolator for road signs."""
    r1 = cv2.inRange(hsv, (0,   60, 40), (14,  255, 255))
    r2 = cv2.inRange(hsv, (166, 60, 40), (180, 255, 255))
    return cv2.bitwise_or(r1, r2)


def _isolate_red_obstacle(hsv) -> np.ndarray:
    """Broad red isolator for 3D red car obstacles and cuboids."""
    r1 = cv2.inRange(hsv, (0,   35, 25), (18,  255, 255))
    r2 = cv2.inRange(hsv, (160, 35, 25), (180, 255, 255))
    return cv2.bitwise_or(r1, r2)


def _isolate_blue(hsv) -> np.ndarray:
    return cv2.inRange(hsv, (100, 90, 50), (135, 255, 255))


def build_obstacle_mask(hsv, sign_boxes=None) -> np.ndarray:
    """
    Masks out 3D red car obstacles on the roadway so their white windshields
    and highlights never corrupt the white/yellow road line masks.
    """
    red_obs = _isolate_red_obstacle(hsv)
    red_obs[:175, :] = 0
    red_obs[455:, :] = 0

    if sign_boxes:
        for (sx, sy, sw, sh) in sign_boxes:
            y1 = max(0, int(sy) - 10)
            y2 = min(480, int(sy + sh) + 10)
            x1 = max(0, int(sx) - 10)
            x2 = min(640, int(sx + sw) + 10)
            red_obs[y1:y2, x1:x2] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    merged = cv2.morphologyEx(red_obs, cv2.MORPH_CLOSE, k_close)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.dilate(merged, k_dil, iterations=1)
    return merged

# =============================================================================
# §9  LANE DETECTION & CONTROLLER (Task 2A Architecture + Asphalt/Grass Check)
# =============================================================================

class LaneDetector:
    """
    Task 2A multi-band lane detector (`360..400` near, `300..340` far) with
    asphalt/grass boundary verification.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.lane               = 'outer'
        self.frames_seen        = 0
        self.prev_e_lat         = 0.0
        self.last_delta         = 0.0
        self.last_cx_near_outer = 230.0
        self.last_cx_far_outer  = 250.0
        self.last_cx_near_inner = 410.0
        self.last_cx_far_inner  = 390.0
        self.last_valid_cx_near = 320.0
        self.last_valid_cx_far  = 320.0
        self.dropout_ctr        = 0

    def detect(self, hsv, active_lane: str = 'outer',
               sign_boxes=None, obstacle_mask=None,
               start_lane: str = None, blend_u: float = 1.0) -> dict:
        if hsv is None or hsv.size == 0:
            self.dropout_ctr += 1
            return {
                'cx_near':       self.last_valid_cx_near,
                'cx_far':        self.last_valid_cx_far,
                'cx_near_outer': self.last_cx_near_outer,
                'cx_far_outer':  self.last_cx_far_outer,
                'cx_near_inner': self.last_cx_near_inner,
                'cx_far_inner':  self.last_cx_far_inner,
                'valid':         False,
                'lane':          active_lane or self.lane or 'outer',
            }

        self.frames_seen += 1
        current_lane = active_lane or self.lane or 'outer'
        self.lane = current_lane

        # Road markings segmentation
        mask_y = cv2.inRange(hsv,
                             np.array([15,  70,  70], dtype=np.uint8),
                             np.array([42, 255, 255], dtype=np.uint8))
        mask_w = cv2.inRange(hsv,
                             np.array([0,   0, 165], dtype=np.uint8),
                             np.array([180, 48, 255], dtype=np.uint8))

        # Road asphalt (dark grey/black) & grass (green) for boundary side verification
        mask_asphalt = cv2.inRange(hsv,
                                   np.array([0,   0,  15], dtype=np.uint8),
                                   np.array([180, 60, 100], dtype=np.uint8))
        mask_grass   = cv2.inRange(hsv,
                                   np.array([38,  50,  40], dtype=np.uint8),
                                   np.array([92, 255, 255], dtype=np.uint8))

        mask_y[:180, :] = 0;  mask_y[450:, :] = 0
        mask_w[:180, :] = 0;  mask_w[450:, :] = 0

        # Blank out traffic signs
        if sign_boxes:
            for (sx, sy, sw, sh) in sign_boxes:
                y1 = max(0, int(sy) - 15);  y2 = min(480, int(sy + sh) + 15)
                x1 = max(0, int(sx) - 15);  x2 = min(640, int(sx + sw) + 15)
                mask_w[y1:y2, x1:x2] = 0
                mask_y[y1:y2, x1:x2] = 0

        # Blank out red obstacles on road
        if obstacle_mask is not None:
            mask_w[obstacle_mask > 0] = 0
            mask_y[obstacle_mask > 0] = 0
            mask_asphalt[obstacle_mask > 0] = 0

        def extract_band_both(r_top: int, r_bot: int, r_mid: float):
            w_half = 140.0 + 0.6 * (r_mid - 280.0)

            # 1. Extract white center divider (must be a narrow dash bordered by dark asphalt)
            w_row_centres = []
            for row in range(r_top, r_bot):
                w_cols = np.where(mask_w[row, :] > 0)[0]
                if len(w_cols) < 3:
                    continue
                segs = [seg for seg in np.split(w_cols, np.where(np.diff(w_cols) > 15)[0] + 1)
                        if 3 <= len(seg) <= 65]
                valid_w_segs = []
                for seg in segs:
                    c1, c2 = int(seg[0]), int(seg[-1])
                    l_asph = float(np.mean(mask_asphalt[row, max(0, c1 - 22):max(1, c1 - 2)] > 0))
                    r_asph = float(np.mean(mask_asphalt[row, min(638, c2 + 3):min(640, c2 + 23)] > 0))
                    if (l_asph >= 0.25 and r_asph >= 0.25) or max(l_asph, r_asph) >= 0.55:
                        valid_w_segs.append(float(np.median(seg)))
                if valid_w_segs:
                    best_w = min(valid_w_segs, key=lambda x: abs(x - 320.0))
                    w_row_centres.append(best_w)

            w_pos = float(np.median(w_row_centres)) if len(w_row_centres) >= 4 else None

            # 2. Extract outer (y_left) and inner (y_right) yellow boundaries
            y_left_rows  = []
            y_right_rows = []

            for row in range(r_top, r_bot):
                y_cols = np.where(mask_y[row, :] > 0)[0]
                if len(y_cols) < 3:
                    continue

                segs = [seg for seg in np.split(y_cols, np.where(np.diff(y_cols) > 15)[0] + 1)
                        if len(seg) >= 3]
                if not segs:
                    continue

                if len(segs) >= 2 and (float(np.median(segs[-1])) - float(np.median(segs[0]))) >= 160.0:
                    y_left_rows.append(float(np.median(segs[0])))
                    y_right_rows.append(float(np.median(segs[-1])))
                elif w_pos is not None:
                    for s in segs:
                        sm = float(np.median(s))
                        if sm < w_pos - 25.0:
                            y_left_rows.append(sm)
                        elif sm > w_pos + 25.0:
                            y_right_rows.append(sm)
                else:
                    s = max(segs, key=len)
                    c1, c2 = int(s[0]), int(s[-1])
                    sm = float(np.median(s))

                    l_asph  = float(np.mean(mask_asphalt[row, max(0, c1 - 28):max(1, c1 - 2)] > 0))
                    r_asph  = float(np.mean(mask_asphalt[row, min(638, c2 + 3):min(640, c2 + 29)] > 0))
                    l_grass = float(np.mean(mask_grass[row, max(0, c1 - 28):max(1, c1 - 2)] > 0))
                    r_grass = float(np.mean(mask_grass[row, min(638, c2 + 3):min(640, c2 + 29)] > 0))

                    road_on_right_score = (r_asph - r_grass) - (l_asph - l_grass)
                    if road_on_right_score > 0.12:
                        y_left_rows.append(sm)
                    elif road_on_right_score < -0.12:
                        y_right_rows.append(sm)
                    else:
                        if current_lane == 'outer':
                            y_left_rows.append(sm)
                        else:
                            y_right_rows.append(sm)

            y_left  = float(np.median(y_left_rows))  if len(y_left_rows)  >= 3 else None
            y_right = float(np.median(y_right_rows)) if len(y_right_rows) >= 3 else None

            # Compute outer lane centre (Task 2A geometry)
            cx_out = None
            if y_left is not None and w_pos is not None and (w_pos - y_left) >= 60.0:
                cx_out = (y_left + w_pos) / 2.0
            elif y_left is not None:
                cx_out = y_left + w_half
            elif w_pos is not None:
                cx_out = w_pos - w_half
            elif y_right is not None:
                cx_out = y_right - 3.0 * w_half

            # Compute inner lane centre (Task 2A geometry)
            cx_inn = None
            if w_pos is not None and y_right is not None and (y_right - w_pos) >= 60.0:
                cx_inn = (w_pos + y_right) / 2.0
            elif y_right is not None:
                cx_inn = y_right - w_half
            elif w_pos is not None:
                cx_inn = w_pos + w_half
            elif y_left is not None:
                cx_inn = y_left + 3.0 * w_half

            return cx_out, cx_inn

        # Task 2A bands: Near (360..400, mid 380) and Far (300..340, mid 320)
        near_out, near_inn = extract_band_both(360, 400, 380.0)
        far_out,  far_inn  = extract_band_both(300, 340, 320.0)

        valid = True
        if near_out is None and far_out is None and near_inn is None and far_inn is None:
            self.dropout_ctr += 1
            valid = False
            near_out, far_out = self.last_cx_near_outer, self.last_cx_far_outer
            near_inn, far_inn = self.last_cx_near_inner, self.last_cx_far_inner
        else:
            self.dropout_ctr = 0
            if near_out is None: near_out = far_out if far_out is not None else self.last_cx_near_outer
            if far_out  is None: far_out  = near_out
            if near_inn is None: near_inn = far_inn if far_inn is not None else self.last_cx_near_inner
            if far_inn  is None: far_inn  = near_inn

            self.last_cx_near_outer, self.last_cx_far_outer = near_out, far_out
            self.last_cx_near_inner, self.last_cx_far_inner = near_inn, far_inn

        target_near = near_out if current_lane == 'outer' else near_inn
        target_far  = far_out  if current_lane == 'outer' else far_inn

        if start_lane is not None and start_lane != current_lane and 0.0 <= blend_u < 1.0:
            start_near = near_out if start_lane == 'outer' else near_inn
            start_far  = far_out  if start_lane == 'outer' else far_inn
            cx_near = (1.0 - blend_u) * start_near + blend_u * target_near
            cx_far  = (1.0 - blend_u) * start_far  + blend_u * target_far
        else:
            cx_near = target_near
            cx_far  = target_far

        self.last_valid_cx_near = cx_near
        self.last_valid_cx_far  = cx_far

        return {
            'cx_near':       float(cx_near),
            'cx_far':        float(cx_far),
            'cx_near_outer': float(near_out),
            'cx_far_outer':  float(far_out),
            'cx_near_inner': float(near_inn),
            'cx_far_inner':  float(far_inn),
            'valid':         valid,
            'lane':          current_lane,
        }

    def compute_control(self, detection: dict,
                        target_base_speed: float,
                        is_changing_lane: bool = False):
        cx_near = max(0.0, min(639.0, float(detection['cx_near'])))
        cx_far  = max(0.0, min(639.0, float(detection['cx_far'])))

        e_lat = (320.0 - cx_near) / 160.0
        dx = cx_far - cx_near
        theta_curve = math.atan2(dx, 120.0)

        d_e = e_lat - self.prev_e_lat
        self.prev_e_lat = e_lat

        k_p     = 1.90
        k_curve = 0.85
        k_d     = 0.08

        raw_delta = k_p * e_lat - k_curve * theta_curve + k_d * d_e

        steer_limit = MAX_LANE_CHANGE_STEER if is_changing_lane else MAX_STEER
        raw_delta = max(-steer_limit, min(steer_limit, raw_delta))

        delta = 0.35 * raw_delta + 0.65 * self.last_delta
        self.last_delta = delta

        steer_ratio = abs(delta) / MAX_STEER
        curve_ratio = min(1.0, abs(theta_curve) / 0.50)
        speed_factor = max(0.45, 1.0 - 0.45 * steer_ratio - 0.25 * curve_ratio)
        speed = max(0.75, target_base_speed * speed_factor)

        return float(delta), float(speed)

# =============================================================================
# §10  SIGN DETECTION (Task 2B Architecture with Strict Building Rejection)
# =============================================================================

def _classify_blue_letter(white_mask: np.ndarray) -> str:
    """Differentiates 'Parking' ('P') from 'Hospital' ('H')."""
    ch, cw = white_mask.shape
    if ch < 6 or cw < 6:
        return "Hospital"

    tr = float(np.mean(white_mask[:ch // 2, cw // 2:] > 0))
    br = float(np.mean(white_mask[ch // 2:, cw // 2:] > 0))
    bl = float(np.mean(white_mask[ch // 2:, :cw // 2] > 0))

    _, w_hier = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    has_hole = False
    if w_hier is not None:
        for hi in w_hier[0]:
            if hi[3] != -1:
                has_hole = True
                break

    ratio_tr = br / (tr + 1e-5)
    ratio_bl = br / (bl + 1e-5)
    if has_hole or ratio_tr < 0.72 or ratio_bl < 0.45:
        return "Parking"
    return "Hospital"


def detect_signs(hsv, white_global) -> list:
    detections = []
    if hsv is None or hsv.size == 0:
        return detections
    H, W = hsv.shape[:2]

    # ── 1. Red signs: Speed40 (circle) or School (triangle) ──────────
    red_mask = _isolate_red(hsv)
    cnts_red, hier_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_red and hier_red is not None:
        for i, c in enumerate(cnts_red):
            area = cv2.contourArea(c)
            if area < 115 or area > 2200:
                continue

            if hier_red[0][i][3] != -1:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if y > H * 0.75 or (y + h) > 285 or w > 85 or h > 85:
                continue

            asp = w / float(h)
            if not (0.68 <= asp <= 1.42):
                continue

            child_idx = hier_red[0][i][2]
            best_child = None
            best_child_area = 0.0
            while child_idx != -1:
                ca = cv2.contourArea(cnts_red[child_idx])
                if ca > best_child_area:
                    best_child_area = ca
                    best_child = cnts_red[child_idx]
                child_idx = hier_red[0][child_idx][0]

            if best_child is None or best_child_area < 0.14 * area:
                continue

            child_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(child_mask, [best_child], -1, 255, -1)
            inner_white = white_global[child_mask > 0]
            if len(inner_white) == 0 or np.mean(inner_white > 0) < 0.20:
                continue

            (_, _), radius = cv2.minEnclosingCircle(c)
            circle_area = math.pi * (radius ** 2)
            circle_ratio = area / (circle_area + 1e-5)

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 3 or circle_ratio < 0.54:
                label = "School"
            else:
                label = "Speed40"

            detections.append({
                "label": label,
                "area":  int(round(float(area))),
                "bbox":  (int(x), int(y), int(w), int(h)),
            })

    # ── 2. Blue signs: Parking or Hospital ───────────────────────────
    blue_mask = _isolate_blue(hsv)
    cnts_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_blue:
        area = cv2.contourArea(c)
        if area < 140 or area > 2200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y > H * 0.75 or (y + h) > 285 or w > 85 or h > 85:
            continue

        asp = w / float(h)
        if not (0.75 <= asp <= 1.35):
            continue

        extent = area / float(w * h)
        if extent < 0.65:
            continue

        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (peri * peri + 1e-5)
        if circ < 0.62:
            continue

        sign_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(sign_mask, [c - np.array([x, y])], -1, 255, -1)
        crop_white = white_global[y:y+h, x:x+w]
        white_in_sign = cv2.bitwise_and(crop_white, crop_white, mask=sign_mask)
        white_ratio = float(np.mean(white_in_sign > 0))
        if not (0.06 <= white_ratio <= 0.50):
            continue

        label = _classify_blue_letter(white_in_sign)
        detections.append({
            "label": label,
            "area":  int(round(float(area))),
            "bbox":  (int(x), int(y), int(w), int(h)),
        })

    return detections

# =============================================================================
# §11  SAME-LANE OBSTACLE DETECTION
# =============================================================================

def detect_obstacle_in_lane(hsv, active_lane: str,
                            lane_det: dict,
                            sign_boxes=None) -> tuple:
    if hsv is None or hsv.size == 0:
        return False, 0

    red_raw = _isolate_red_obstacle(hsv)
    red_raw[:170, :] = 0
    red_raw[450:, :] = 0

    if sign_boxes:
        for (sx, sy, sw, sh) in sign_boxes:
            y1 = max(0, int(sy) - 10);  y2 = min(480, int(sy + sh) + 10)
            x1 = max(0, int(sx) - 10);  x2 = min(640, int(sx + sw) + 10)
            red_raw[y1:y2, x1:x2] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    red_cl  = cv2.morphologyEx(red_raw, cv2.MORPH_CLOSE, k_close)
    k_dil   = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    red_cl  = cv2.dilate(red_cl, k_dil, iterations=1)

    cnts, _ = cv2.findContours(red_cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, 0

    near_out = lane_det.get('cx_near_outer', 230.0)
    far_out  = lane_det.get('cx_far_outer',  250.0)
    near_inn = lane_det.get('cx_near_inner', 410.0)
    far_inn  = lane_det.get('cx_far_inner',  390.0)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (2000 <= area <= 16500):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        y_bot = y + h
        if not (205 <= y_bot <= 435):
            continue

        ar = w / float(h)
        if not (0.35 <= ar <= 3.5):
            continue

        raw_px = int(np.sum(red_raw[y:y+h, x:x+w] > 0))
        if raw_px < 80:
            continue

        obs_cx = x + w / 2.0

        y_clamped = min(410.0, max(260.0, float(y_bot)))
        alpha = (y_clamped - 320.0) / (380.0 - 320.0)
        cx_out_at_obs = far_out + alpha * (near_out - far_out)
        cx_inn_at_obs = far_inn + alpha * (near_inn - far_inn)
        divider_x     = 0.5 * (cx_out_at_obs + cx_inn_at_obs)

        if obs_cx < (cx_out_at_obs - 140.0) or obs_cx > (cx_inn_at_obs + 140.0):
            continue

        obs_lane = 'outer' if obs_cx < divider_x else 'inner'
        if obs_lane != active_lane:
            continue

        candidates.append((area, x, y, w, h))

    if not candidates:
        return False, 0

    candidates.sort(key=lambda t: t[0], reverse=True)
    best = candidates[0]
    return True, int(round(float(best[0])))

# =============================================================================
# §12  ARUCO DETECTION & POSE ESTIMATION
# =============================================================================

def find_marker_corners(frame, target_id: int = TARGET_BAY_ID):
    """
    Strictly matches `target_id` (ID 1) at distances up to ~1.10 m so the
    parking S-curve starts early right after exiting the final corner.
    """
    if frame is None or frame.size == 0:
        return None, None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        try:
            params = cv2.aruco.DetectorParameters()
            params.minMarkerPerimeterRate = 0.025
            detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
            corners, ids, _ = detector.detectMarkers(gray)
        except (AttributeError, Exception):
            params = cv2.aruco.DetectorParameters_create()
            params.minMarkerPerimeterRate = 0.025
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, cv2.aruco.Dictionary_get(ARUCO_DICT), parameters=params)

        if ids is None or len(np.asarray(ids).ravel()) == 0:
            return None, None

        ids_flat = np.asarray(ids).ravel()
        for i, mid in enumerate(ids_flat):
            if int(mid) == int(target_id):
                pts = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] >= 4:
                    return pts[:4], int(mid)

        return None, None
    except Exception:
        return None, None


class PoseEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.heading = 0.0
        self.rho     = None

    @staticmethod
    def _bearing(corners: np.ndarray) -> float:
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())
        return float(math.atan2(
            (cx - IMG_W / 2.0) / FOCAL_PX,
            math.cos(CAM_PITCH) + ((IMG_H / 2.0 - cy) / FOCAL_PX) * math.sin(CAM_PITCH)))

    @staticmethod
    def _pnp(corners: np.ndarray):
        try:
            _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                MARKER_CORNERS_3D, corners.astype(np.float32),
                CAMERA_MATRIX, np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            sols = []
            for rv, tv, e in zip(rvecs, tvecs, np.array(errs).flatten()):
                R, _ = cv2.Rodrigues(rv)
                fwd  = R.T @ np.array([0.0, 0.0, 1.0])
                sols.append((float(e), float(math.atan2(fwd[0], -fwd[2])), tv.flatten()))
            sols.sort(key=lambda s: s[0])
            t   = sols[0][2]
            rho = float(math.sqrt(max(float(np.dot(t, t)) - CAM_ABOVE_MK ** 2, 1e-6)))
            clear = len(sols) < 2 or sols[1][0] > AMBIGUITY_RATIO * max(sols[0][0], 1e-6)
            return (sols[0][1] if clear else None), rho
        except Exception:
            return None, 1.0

    def update(self, corners, last_steering: float):
        if corners is None:
            return None
        bearing           = self._bearing(corners)
        yaw_seen, rho_now = self._pnp(corners)
        if self.rho is None:
            self.rho = rho_now
            if yaw_seen is not None and abs(yaw_seen) < math.radians(30.0):
                self.heading = 0.5 * yaw_seen
        rho_prev, self.rho = self.rho, 0.65 * self.rho + 0.35 * rho_now

        ds = min(max((rho_prev - self.rho) / max(math.cos(bearing), 0.3), 0.0), 0.008)
        self.heading -= math.tan(last_steering) / WHEELBASE * ds
        if yaw_seen is not None and abs(yaw_seen - self.heading) < YAW_GATE:
            self.heading += YAW_BLEND * (yaw_seen - self.heading)

        los           = bearing + self.heading
        lateral_front = -(-self.rho * math.sin(los)
                          + (WHEELBASE / 2.0 - CAM_AHEAD) * math.sin(self.heading))
        forward       = self.rho * math.cos(los) + CAM_AHEAD * math.cos(self.heading)
        return {
            'lateral': float(lateral_front),
            'heading': float(-self.heading),
            'forward': float(forward),
            'bearing': float(-bearing),
        }

# =============================================================================
# §13  PARKING CONTROLLER (Adaptive Runway for Inner/Outer Lane Entry)
# =============================================================================

class ParkController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.initialized   = False
        self.parked        = False
        self.is_bay_1      = True
        self.d_start       = 1.0
        self.l_start       = 0.35
        self.d_merge       = 0.32
        self.last_pose     = {
            'lateral': 0.35,
            'heading': 0.0,
            'forward': 1.0,
            'bearing': 0.0,
        }
        self.prev_heading  = 0.0
        self.last_steering = 0.0
        self.last_speed    = 0.0
        self.none_count    = 0
        self.seen_frames   = 0

    def step(self, pose):
        if self.parked:
            return 0.0, 0.0, True

        if pose is None:
            self.none_count += 1
            if not self.initialized or self.seen_frames < 12 or self.last_pose['forward'] > 0.32:
                return 0.0, 0.0, False
            est_forward = self.last_pose['forward'] - 0.0025
            self.last_pose['forward'] = max(BAY_MIDDLE_FORWARD - 0.01, est_forward)
            curr_pose = self.last_pose
        else:
            self.none_count  = 0
            self.seen_frames += 1
            self.last_pose   = dict(pose)
            curr_pose        = pose

        lateral = float(curr_pose.get('lateral', 0.0))
        heading = float(curr_pose.get('heading', 0.0))
        forward = float(curr_pose.get('forward', 1.0))

        # Stop when vehicle reaches the middle of the bay
        if forward <= (BAY_MIDDLE_FORWARD + 0.005) and self.seen_frames >= 12:
            self.parked = True
            self.last_speed = 0.0
            self.last_steering = 0.0
            return 0.0, 0.0, True

        if not self.initialized:
            self.l_start  = lateral
            self.is_bay_1 = (TARGET_BAY_ID == 1) or bool(lateral > 0.28)
            # Dynamically guarantee at least 0.35 m of S-curve runway even if the
            # marker is acquired close to the bay after rounding the bottom corner
            self.d_merge  = min(BAY_MOUTH_FORWARD + 0.10, max(0.24, forward - 0.35))
            self.d_start  = max(self.d_merge + 0.22, forward)
            self.initialized = True
            print(f"[PARKING INIT] d_start={self.d_start:.2f}m, l_start={self.l_start:.2f}m, "
                  f"d_merge={self.d_merge:.2f}m")

        lead_bias = -0.022 if self.is_bay_1 else 0.0

        # ZONE 1: S-Curve Transition into the Bay
        if forward > self.d_merge:
            runway   = max(0.18, self.d_start - self.d_merge)
            u        = max(0.0, min(1.0, (self.d_start - forward) / runway))
            y_ref    = lead_bias + ((self.l_start - lead_bias) / 2.0) * (1.0 + math.cos(math.pi * u))
            dy_dx    = -((math.pi * (self.l_start - lead_bias)) / (2.0 * runway)) * math.sin(math.pi * u)
            psi_ref  = max(-0.70, min(0.70, math.atan(dy_dx)))

            err_lat  = lateral - y_ref
            err_head = heading - psi_ref
            d_head   = heading - self.prev_heading

            raw_steer = (-1.15 * err_head) - math.atan(3.5 * err_lat) - (0.20 * d_head)
            steering  = max(-MAX_STEER, min(MAX_STEER, raw_steer))
            steer_ratio = abs(steering) / MAX_STEER
            speed     = 0.95 * (1.0 - 0.25 * steer_ratio)

        # ZONE 2: Bay Mouth Glide-In
        elif forward > BAY_MOUTH_FORWARD:
            fade     = (forward - BAY_MOUTH_FORWARD) / max(0.01, self.d_merge - BAY_MOUTH_FORWARD)
            y_target = lead_bias * max(0.0, min(1.0, fade))

            err_lat  = lateral - y_target
            err_head = heading
            d_head   = heading - self.prev_heading

            raw_steer = (-1.60 * err_head) - (2.80 * err_lat) - (0.25 * d_head)
            lim       = 0.45 if abs(lateral) > 0.06 else 0.25
            steering  = max(-lim, min(lim, raw_steer))
            speed     = 0.55

        # ZONE 3: Inside Bay Creep to Stop Target
        else:
            err_lat  = lateral
            err_head = heading
            d_head   = heading - self.prev_heading

            raw_steer = (-1.80 * err_head) - (3.20 * err_lat) - (0.05 * d_head)
            # Keep steering authority if still pulling laterally into the bay
            lim       = 0.40 if abs(lateral) > 0.05 else MAX_STEER_BAY
            steering  = max(-lim, min(lim, raw_steer))
            dist_to_stop = max(0.0, forward - BAY_MIDDLE_FORWARD)
            speed     = max(0.25, min(PARKING_DOCK_SPEED, 0.25 + 1.20 * dist_to_stop))

        self.prev_heading  = heading
        self.last_steering = steering
        self.last_speed    = speed
        return float(steering), float(speed), False

# =============================================================================
# §14  MAIN CONTROL LOOP
# =============================================================================

def run_task3():
    print("=" * 65)
    print("  eYRC Niti Vahan — Task 3   Team eYRC#3576")
    print("=" * 65)
    client = RemoteAPIClient()
    sim    = (client.getObject('sim')
              if hasattr(client, 'getObject') else client.require('sim'))

    handles: dict = {}
    for name, candidates in [
        ('vision_sensor', VISION_SENSOR_CANDIDATES),
        ('steer_left',    STEER_LEFT_CANDIDATES),
        ('steer_right',   STEER_RIGHT_CANDIDATES),
        ('motor_left',    MOTOR_LEFT_CANDIDATES),
        ('motor_right',   MOTOR_RIGHT_CANDIDATES),
    ]:
        for path in candidates:
            try:
                h = (sim.getObject(path, {'noError': True})
                     if hasattr(sim, 'getObject') else sim.getObject(path))
                if h != -1:
                    handles[name] = h
                    print(f"[INIT] {name:14s} resolved: '{path}' (Handle {h})")
                    break
            except Exception:
                pass

    simulation_started_here = False
    if sim.getSimulationState() in (sim.simulation_stopped, sim.simulation_paused):
        sim.startSimulation()
        simulation_started_here = True

    while sim.getSimulationState() not in (
            sim.simulation_advancing_running,
            sim.simulation_advancing_firstafterstop):
        time.sleep(0.05)

    print("\n" + "=" * 65)
    print(">>> NITI VAHAN AUTONOMOUS MISSION RUNNING <<<")
    print(f"    Base Cruise Speed  : {BASE_CRUISE_SPEED:.2f} rad/s")
    print(f"    Regulated Speed    : {SPEED_ZONE_SPEED:.2f} rad/s")
    print("=" * 65 + "\n")

    lane_detector   = LaneDetector()
    pose_estimator  = PoseEstimator()
    park_controller = ParkController()

    state       = RobotState.LANE_FOLLOWING
    active_lane = 'outer'

    has_halted_at_hospital   = False
    hospital_halt_start_time = 0.0
    last_hospital_time       = -999.0

    in_speed_zone            = False
    speed_zone_completed     = False
    last_speed_sign_time     = -999.0
    speed_zone_exit_time     = -999.0
    speed_sign_armed         = True
    speed_sign_absent_frames = 0

    has_seen_school          = False
    parking_zone_active      = False
    last_parking_sign_time   = -999.0
    aruco_confirm_frames     = 0

    obstacles_avoided        = 0
    last_obstacle_time       = -999.0
    obstacle_armed           = True
    obstacle_absent_frames   = 0
    lane_change_start_lane   = 'outer'
    lane_change_target_lane  = 'inner'
    lane_change_start_time   = 0.0
    obstacle_detected_frames = 0

    frame_counter            = 0

    try:
        while sim.getSimulationState() in (
                sim.simulation_advancing_running,
                sim.simulation_advancing_firstafterstop):

            current_sim_time = sim.getSimulationTime()

            frame, hsv = read_frame(sim, handles['vision_sensor'])
            if frame is None or hsv is None:
                continue
            frame_counter += 1

            white_global = cv2.inRange(hsv, (0, 0, 130), (180, 60, 255))
            raw_signs    = detect_signs(hsv, white_global)
            sign_boxes   = [s['bbox'] for s in raw_signs if 'bbox' in s]

            # Mask out red obstacles on road so their windshields never corrupt lane lines
            obs_mask = build_obstacle_mask(hsv, sign_boxes=sign_boxes)

            # Filter out any red sign detection whose bounding box overlaps an on-road obstacle
            filtered_signs = []
            for s in raw_signs:
                sx, sy, sw, sh = s['bbox']
                if s['label'] in ('Speed40', 'School'):
                    roi_obs = obs_mask[max(0, sy):min(480, sy + sh), max(0, sx):min(640, sx + sw)]
                    if roi_obs.size > 0 and float(np.mean(roi_obs > 0)) > 0.30:
                        continue
                filtered_signs.append(s)

            # Suppress ONLY Hospital sign re-detection while departing the Hospital
            if has_halted_at_hospital and (
                    current_sim_time - (hospital_halt_start_time + HOSPITAL_HALT_DURATION)
                    < HOSPITAL_SUPPRESS_WINDOW):
                signs = [s for s in filtered_signs if s['label'] != 'Hospital']
            else:
                signs = filtered_signs

            # ==========================================================
            # STATE: HALTING_HOSPITAL
            # ==========================================================
            if state == RobotState.HALTING_HOSPITAL:
                stop_vehicle(sim, handles)
                elapsed = current_sim_time - hospital_halt_start_time
                if elapsed >= HOSPITAL_HALT_DURATION:
                    has_halted_at_hospital = True
                    state = RobotState.LANE_FOLLOWING
                    print(f"[HOSPITAL] {HOSPITAL_HALT_DURATION:.1f}s halt completed "
                          f"at Sim Time {current_sim_time:.2f}s! Resuming cruise.")
                continue

            # ==========================================================
            # STATE: PARKING_MANEUVER (Task 2C Benchmark Controller)
            # ==========================================================
            elif state == RobotState.PARKING_MANEUVER:
                corners, _ = find_marker_corners(frame, TARGET_BAY_ID)
                pose       = pose_estimator.update(corners, park_controller.last_steering)

                # If marker is lost before entering the bay mouth, resume lane following
                if pose is None and park_controller.last_pose['forward'] > 0.32:
                    if park_controller.none_count > 15:
                        print("[PARKING] Marker lost before bay mouth — resuming lane tracking.")
                        park_controller.reset()
                        pose_estimator.reset()
                        aruco_confirm_frames = 0
                        state = RobotState.LANE_FOLLOWING
                        continue

                steering, speed, is_parked = park_controller.step(pose)

                if is_parked:
                    stop_vehicle(sim, handles)
                    state = RobotState.PARKED
                    print("\n" + "=" * 55)
                    print(">>> TASK 3 MISSION SUCCESSFULLY COMPLETED <<<")
                    print(f"- Hospital Halt     : {has_halted_at_hospital}")
                    print(f"- Speed 40 Zone     : {speed_zone_completed}")
                    print(f"- Obstacles Avoided : {obstacles_avoided}")
                    print(f"- Bay Docking       : SUCCESS")
                    print("=" * 55 + "\n")
                    time.sleep(0.8)
                    sim.stopSimulation()
                    break
                else:
                    drive(sim, handles, steering, speed)
                continue

            # ==========================================================
            # TRAFFIC SIGN PERCEPTION (Active in both LANE_FOLLOWING & LANE_CHANGING)
            # ==========================================================
            if any(s['label'] == 'Speed40' and s['area'] >= SPEED_SIGN_MIN_AREA
                   for s in signs):
                speed_sign_absent_frames = 0
            else:
                speed_sign_absent_frames += 1
                if speed_sign_absent_frames >= SPEED_SIGN_REARM_FRAMES:
                    speed_sign_armed = True

            for s in signs:
                label = s['label']
                area  = s['area']

                if (label == 'Hospital'
                        and not has_halted_at_hospital
                        and area >= 150
                        and (current_sim_time - last_hospital_time) > 3.0):
                    hospital_halt_start_time = current_sim_time
                    last_hospital_time       = current_sim_time
                    state                    = RobotState.HALTING_HOSPITAL
                    stop_vehicle(sim, handles)
                    print(f"\n[TRAFFIC RULE - HOSPITAL] Halting for "
                          f"{HOSPITAL_HALT_DURATION:.1f}s at Sim Time {current_sim_time:.2f}s.")
                    break

                elif label == 'Speed40':
                    if (not in_speed_zone and not speed_zone_completed
                            and speed_sign_armed and area >= SPEED_SIGN_MIN_AREA):
                        in_speed_zone            = True
                        speed_sign_armed         = False
                        speed_sign_absent_frames = 0
                        last_speed_sign_time     = current_sim_time
                        print(f"\n[TRAFFIC RULE - SPEED ZONE] Speed 40 sign detected "
                              f"(Area: {area})! Regulating speed to {SPEED_ZONE_SPEED} rad/s.")
                    elif (in_speed_zone and speed_sign_armed
                            and area >= SPEED_SIGN_MIN_AREA
                            and (current_sim_time - last_speed_sign_time)
                            >= SPEED_SIGN_COOLDOWN_SECONDS):
                        in_speed_zone            = False
                        speed_zone_completed     = True
                        speed_sign_armed         = False
                        speed_sign_absent_frames = 0
                        speed_zone_exit_time     = current_sim_time
                        print(f"\n[TRAFFIC RULE - SPEED ZONE] End of Speed 40 zone "
                              f"(Area: {area})! Resuming cruise speed {BASE_CRUISE_SPEED} rad/s.")

                elif label == 'School':
                    if not has_seen_school and area >= 115:
                        has_seen_school = True
                        print(f"\n[TRAFFIC RULE - SCHOOL] School sign detected "
                              f"(Area: {area}) — proceeding safely.")

                elif (label == 'Parking'
                        and has_halted_at_hospital
                        and speed_zone_completed
                        and (current_sim_time - speed_zone_exit_time) >= 45.0
                        and not parking_zone_active
                        and area >= 140
                        and (current_sim_time - last_parking_sign_time) > 3.0):
                    last_parking_sign_time = current_sim_time
                    parking_zone_active    = True
                    print(f"\n[TRAFFIC RULE - PARKING] Parking sign detected (Area: {area})! "
                          f"Maintaining '{active_lane}' lane and scanning for ArUco ID_1.")

            if state == RobotState.HALTING_HOSPITAL:
                continue

            # ==========================================================
            # ARUCO BAY MARKER CHECK
            # Only active after driving down the right-hand straight (>= 65s after
            # exiting Speed 40 zone) and when not mid-lane-change.
            # ==========================================================
            ready_for_parking_search = (
                has_halted_at_hospital
                and speed_zone_completed
                and state == RobotState.LANE_FOLLOWING
                and (parking_zone_active
                     or (current_sim_time - speed_zone_exit_time) >= PARKING_SEARCH_DELAY)
                and (current_sim_time - last_obstacle_time) >= 10.0
            )

            if ready_for_parking_search:
                corners, marker_id = find_marker_corners(frame, TARGET_BAY_ID)
                valid_bay_marker = False
                if corners is not None:
                    pts     = np.asarray(corners, dtype=np.float32)
                    mk_area = float(cv2.contourArea(pts))
                    mk_cy   = float(pts[:, 1].mean())
                    # Lock onto ID_1 from up to ~1.05 m away (area >= 48 px^2)
                    if mk_area >= 48.0 and 100.0 <= mk_cy <= 360.0:
                        _, rho_est = PoseEstimator._pnp(pts)
                        brg_est    = PoseEstimator._bearing(pts)
                        fwd_est    = rho_est * math.cos(brg_est) + CAM_AHEAD
                        lat_est    = rho_est * math.sin(brg_est)
                        # Real parking bay is ahead (0.30..1.12 m) and to the right
                        if 0.30 <= fwd_est <= 1.12 and 0.04 <= lat_est <= 0.68:
                            valid_bay_marker = True

                if valid_bay_marker:
                    aruco_confirm_frames += 1
                    if aruco_confirm_frames >= ARUCO_CONFIRM_FRAMES:
                        aruco_confirm_frames = 0
                        pose_estimator.reset()
                        park_controller.reset()
                        pose = pose_estimator.update(corners, lane_detector.last_delta)
                        park_controller.step(pose)
                        state = RobotState.PARKING_MANEUVER
                        print(f"\n[PARKING] ArUco ID_{marker_id} confirmed — engaging precision docking.")
                        continue
                else:
                    aruco_confirm_frames = 0

            # ==========================================================
            # STATE: LANE_CHANGING (Triggered ONLY by Obstacles)
            # ==========================================================
            if state == RobotState.LANE_CHANGING:
                elapsed  = max(0.0, current_sim_time - lane_change_start_time)
                progress = min(1.0, elapsed / max(LANE_CHANGE_DURATION, 1e-3))

                # Cosine S-curve blend from start_lane (u=0) to target_lane (u=1)
                u = 0.5 * (1.0 - math.cos(math.pi * progress))

                detection = lane_detector.detect(
                    hsv,
                    active_lane=lane_change_target_lane,
                    sign_boxes=sign_boxes,
                    obstacle_mask=obs_mask,
                    start_lane=lane_change_start_lane,
                    blend_u=u,
                )
                target_speed = min(
                    LANE_CHANGE_SPEED,
                    SPEED_ZONE_SPEED if in_speed_zone else BASE_CRUISE_SPEED
                )
                delta, speed = lane_detector.compute_control(
                    detection, target_speed, is_changing_lane=True
                )
                drive(sim, handles, delta, speed)

                if elapsed >= LANE_CHANGE_DURATION:
                    active_lane = lane_change_target_lane
                    state       = RobotState.LANE_FOLLOWING
                    print(f"[LANE CHANGE] Completed smoothly! Retained '{active_lane}' lane.")
                continue

            # ==========================================================
            # STATE: LANE_FOLLOWING — Obstacle Check & Normal Cruise
            # ==========================================================
            detection = lane_detector.detect(
                hsv, active_lane=active_lane, sign_boxes=sign_boxes, obstacle_mask=obs_mask)

            obs_in_lane, obs_area = detect_obstacle_in_lane(
                hsv, active_lane=active_lane, lane_det=detection, sign_boxes=sign_boxes)

            if not obstacle_armed:
                if obs_in_lane:
                    obstacle_absent_frames = 0
                else:
                    obstacle_absent_frames += 1

                if (obstacle_absent_frames >= OBSTACLE_REARM_FRAMES
                        and (current_sim_time - last_obstacle_time) >= OBSTACLE_COOLDOWN_TIME):
                    obstacle_armed           = True
                    obstacle_detected_frames = 0
            else:
                if obs_in_lane:
                    obstacle_detected_frames += 1
                else:
                    obstacle_detected_frames = 0

                if obstacle_detected_frames >= OBSTACLE_CONFIRM_FRAMES:
                    obstacle_detected_frames = 0
                    obstacle_armed           = False
                    obstacle_absent_frames   = 0
                    lane_change_start_lane   = active_lane
                    lane_change_target_lane  = 'inner' if active_lane == 'outer' else 'outer'
                    active_lane              = lane_change_target_lane
                    obstacles_avoided       += 1
                    state                    = RobotState.LANE_CHANGING
                    lane_change_start_time   = current_sim_time
                    last_obstacle_time       = current_sim_time
                    print(f"\n[OBSTACLE #{obstacles_avoided}] Obstacle confirmed in "
                          f"'{lane_change_start_lane}' lane (Area: {obs_area})! "
                          f"Smoothly shifting to '{lane_change_target_lane}' lane.")
                    continue

            # Automatically slow down as the vehicle rounds the final corner toward the parking bays
            approaching_parking_corner = (
                speed_zone_completed
                and (current_sim_time - speed_zone_exit_time) >= 75.0
            )
            if parking_zone_active or approaching_parking_corner:
                target_speed = PARKING_APPROACH_SPEED
            elif in_speed_zone:
                target_speed = SPEED_ZONE_SPEED
            else:
                target_speed = BASE_CRUISE_SPEED

            delta, speed = lane_detector.compute_control(
                detection, target_speed, is_changing_lane=False)
            drive(sim, handles, delta, speed)

    except KeyboardInterrupt:
        print("\n[USER] KeyboardInterrupt — stopping vehicle.")
    except Exception as e:
        import traceback
        print(f"\n[EXCEPTION] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        stop_vehicle(sim, handles)
        if simulation_started_here:
            try:
                sim.stopSimulation()
            except Exception:
                pass
        print("[SYSTEM] Simulation stopped.")


if __name__ == '__main__':
    run_task3()