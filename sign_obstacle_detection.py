'''
*****************************************************************************************
*
-  ===============================================
-     Niti Vahan (NV) Theme of eYRC 2026-27
-  ===============================================
*
-  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
-  Filename:         sign_obstacle_detection.py
-  Created:          2026
-  Last Modified:
-  Author:           e-Yantra Team
*
-  You are ONLY allowed to write your code inside the block marked
-  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
-  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          e#YRC3576
# Author List:      Pankaj Amrate, Karan Singh , Krishna Sharma , Prasoon Dhakad
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    '''
    Purpose:
    ---
    Detect every road sign visible in a single image, and report what each
    one is and how large it appears. An image can hold zero, one, or more
    than one sign at once.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `detections` :  [ list of dict ]
        Zero, one, or many entries - one per sign found in the image:
        [
            {
                "label" : str,   one of SIGN_LABELS
                "area"  : int,   that sign's area in pixels
            },
            ...
        ]

    Example call:
    ---
    detections = detect_signs(image)
    '''
    detections = []
    if image is None:
        return detections

    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Red signs: Speed40 (circular ring) or School (triangular border)
    red_mask = _isolate_red(hsv)
    cnts_red, hier_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_red and hier_red is not None:
        for i, c in enumerate(cnts_red):
            area = cv2.contourArea(c)
            if area < 150:
                continue

            # Outer boundary check (parent == -1)
            if hier_red[0][i][3] != -1:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if y > H * 0.85 or w > W * 0.40 or h > H * 0.40:
                continue

            # Sign must have a white interior containing the symbol
            crop_hsv = hsv[y:y+h, x:x+w]
            if not _has_white_symbol(crop_hsv, min_ratio=0.08):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            circ = 4.0 * np.pi * area / (peri * peri + 1e-5)

            if len(approx) == 3 or circ < 0.65:
                detections.append({"label": "School", "area": int(round(float(area)))})
            else:
                detections.append({"label": "Speed40", "area": int(round(float(area)))})

    # 2. Yellow signs: School (supports yellow background boards)
    yellow_mask = _isolate_yellow(hsv)
    cnts_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_yellow:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y > H * 0.85 or w > W * 0.35 or h > H * 0.35:
            continue
        asp = w / float(h)
        if not (0.70 <= asp <= 1.40):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        circ = 4.0 * np.pi * area / (peri * peri + 1e-5)
        extent = area / float(w * h)
        if extent > 0.40 and (len(approx) in (3, 4) or circ > 0.45):
            detections.append({"label": "School", "area": int(round(float(area)))})

    # 3. Blue signs: Parking or Hospital
    blue_mask = _isolate_blue(hsv)
    cnts_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_blue:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y > H * 0.85 or w > W * 0.30 or h > H * 0.30:
            continue

        asp = w / float(h)
        if not (0.80 <= asp <= 1.30):
            continue

        extent = area / float(w * h)
        if extent < 0.65:
            continue

        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (peri * peri + 1e-5)
        if circ < 0.65:
            continue

        # Must contain white letter symbol (P or H)
        crop_hsv = hsv[y:y+h, x:x+w]
        white_mask = cv2.inRange(crop_hsv, (0, 0, 150), (180, 60, 255))
        if np.mean(white_mask > 0) < 0.08:
            continue

        label = _classify_blue_letter(white_mask)
        detections.append({"label": label, "area": int(round(float(area)))})

    return detections


def detect_obstacle(image):
    '''
    Purpose:
    ---
    Detect whether the obstacle is visible in a single image, and how large
    it appears.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "present" : bool,   True if the obstacle is visible
            "area"    : int,    its area in pixels, 0 if not present
        }

    Example call:
    ---
    result = detect_obstacle(image)
    '''
    if image is None:
        return _ObstacleResult(False, 0)

    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Candidates: Red cuboid block, Cyan block, or Magenta block
    red_mask = _isolate_red(hsv)
    cyan_mask = cv2.inRange(hsv, (80, 100, 50), (105, 255, 255))
    mag_mask = cv2.inRange(hsv, (140, 100, 50), (165, 255, 255))

    candidates = []
    for mask in [red_mask, cyan_mask, mag_mask]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w / 2.0

            # Obstacle sits directly on the road / center columns
            if not (0.05 * W <= cx <= 0.95 * W):
                continue
            asp = w / float(h)
            if not (0.65 <= asp <= 1.50):
                continue
            extent = area / float(w * h)
            if extent < 0.60:
                continue

            # Must be a solid obstacle block without hollow white center
            crop_hsv = hsv[y:y+h, x:x+w]
            white_mask = cv2.inRange(crop_hsv, (0, 0, 150), (180, 60, 255))
            if np.mean(white_mask > 0) > 0.25:
                continue

            candidates.append((area, c))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return _ObstacleResult(True, int(round(float(candidates[0][0]))))

    return _ObstacleResult(False, 0)


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_signs() or detect_obstacle() - the
# evaluation script only ever calls those two functions. List them in
# the file header too.
# ------------------------------------------------------------------

class _ObstacleResult(dict):
    '''
    A dict subclass supporting both dict access/type verification
    and tuple unpacking (present, area).
    '''
    def __init__(self, present, area):
        super().__init__({"present": bool(present), "area": int(area)})

    def __iter__(self):
        yield self["present"]
        yield self["area"]

    def __getitem__(self, key):
        if key == 0:
            return self["present"]
        if key == 1:
            return self["area"]
        return super().__getitem__(key)


def _isolate_red(hsv):
    '''Isolate Red color with hue wrap-around handling.'''
    r1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    r2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    return cv2.bitwise_or(r1, r2)


def _isolate_blue(hsv):
    '''Isolate Blue color for Parking and Hospital signs.'''
    return cv2.inRange(hsv, (100, 100, 50), (135, 255, 255))


def _isolate_yellow(hsv):
    '''Isolate Yellow color.'''
    return cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))


def _has_white_symbol(crop_hsv, min_ratio=0.08):
    '''Check if a sign candidate has a bright/white interior symbol.'''
    white_mask = cv2.inRange(crop_hsv, (0, 0, 150), (180, 60, 255))
    return bool(np.mean(white_mask > 0) >= min_ratio)


def _classify_blue_letter(white_mask):
    '''
    Differentiate between Parking (P) and Hospital (H) using structural
    and topological analysis of the white symbol:
- Letter "P" has an enclosed loop/hole in the upper half and an empty
      bottom-right quadrant.
- Letter "H" is vertically and horizontally symmetric with no inner loop.
    '''
    ch, cw = white_mask.shape
    tr = np.mean(white_mask[:ch//2, cw//2:] > 0)
    br = np.mean(white_mask[ch//2:, cw//2:] > 0)
    w_cnts, w_hier = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    has_hole = False
    if w_hier is not None:
        for hi in w_hier[0]:
            if hi[3] != -1:
                has_hole = True
                break
    if has_hole or (br < 0.72 * tr):
        return "Parking"
    return "Hospital"


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area