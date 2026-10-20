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
# Functions:        detect_signs, detect_obstacle, _isolate_red, _isolate_blue, _classify_blue_letter
# Global variables: None


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
    Detect every road sign visible in a single image.
    '''
    detections = []
    if image is None or image.size == 0:
        return detections

    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_global = cv2.inRange(hsv, (0, 0, 130), (180, 60, 255))

    # 1. Red signs: Speed40 (circular ring) or School (triangular border)
    red_mask = _isolate_red(hsv)
    cnts_red, hier_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_red and hier_red is not None:
        for i, c in enumerate(cnts_red):
            area = cv2.contourArea(c)
            if area < 150:
                continue

            # Must be an outer contour (parent == -1)
            if hier_red[0][i][3] != -1:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if y > H * 0.85 or w > W * 0.40 or h > H * 0.40:
                continue

            asp = w / float(h)
            if not (0.70 <= asp <= 1.40):
                continue

            # Real signs are hollow borders (annulus / triangle frame) enclosing
            # a white background. They MUST have an inner child contour (hole).
            child_idx = hier_red[0][i][2]
            if child_idx == -1:
                continue

            child_area = cv2.contourArea(cnts_red[child_idx])
            if child_area < 0.15 * area:
                continue

            # Check that the interior inside the child contour is actually white
            child_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(child_mask, [cnts_red[child_idx]], -1, 255, -1)
            inner_white = white_global[child_mask > 0]
            if len(inner_white) == 0 or np.mean(inner_white > 0) < 0.20:
                continue

            # Classify circle (Speed40) vs triangle (School)
            (cx_c, cy_c), radius = cv2.minEnclosingCircle(c)
            circle_area = np.pi * (radius ** 2)
            circle_ratio = area / (circle_area + 1e-5)

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) == 3 or circle_ratio < 0.52:
                label = "School"
            else:
                label = "Speed40"

            detections.append({"label": label, "area": int(round(float(area)))})

    # 2. Blue signs: Parking or Hospital
    blue_mask = _isolate_blue(hsv)
    cnts_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_blue:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y > H * 0.85 or w > W * 0.35 or h > H * 0.35:
            continue

        asp = w / float(h)
        if not (0.75 <= asp <= 1.35):
            continue

        extent = area / float(w * h)
        if extent < 0.60:
            continue

        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (peri * peri + 1e-5)
        if circ < 0.60:
            continue

        # Must contain white letter symbol (P or H) inside the blue sign contour
        sign_mask = np.zeros((h, w), dtype=np.uint8)
        c_shifted = c - [x, y]
        cv2.drawContours(sign_mask, [c_shifted], -1, 255, -1)

        crop_white = white_global[y:y+h, x:x+w]
        white_in_sign = cv2.bitwise_and(crop_white, crop_white, mask=sign_mask)
        if np.mean(white_in_sign > 0) < 0.06:
            continue

        label = _classify_blue_letter(white_in_sign)
        detections.append({"label": label, "area": int(round(float(area)))})

    return detections


def detect_obstacle(image):
    '''
    Detect whether the obstacle is visible in a single image.
    '''
    if image is None or image.size == 0:
        return {"present": False, "area": 0}

    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_global = cv2.inRange(hsv, (0, 0, 130), (180, 60, 255))

    # 1. Isolate the obstacle's colour (red cuboid)
    red_mask = _isolate_red(hsv)
    cnts, hier = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or hier is None:
        return {"present": False, "area": 0}

    candidates = []
    for i, c in enumerate(cnts):
        # Outer boundary only
        if hier[0][i][3] != -1:
            continue

        area = cv2.contourArea(c)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Near the top of the image (y < 0.08*H), roadway is only near center
        if y < 0.08 * H and (cx < 0.30 * W or cx > 0.70 * W):
            continue

        # Roadway bounds
        if not (0.05 * W <= cx <= 0.95 * W):
            continue

        aspect_ratio = w / float(h)
        if not (0.45 <= aspect_ratio <= 2.50):
            continue

        extent = area / float(w * h)
        if extent < 0.45:
            continue

        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / (peri * peri + 1e-5)
        if circ < 0.35:
            continue

        # 2. Tell it apart from red signs:
        # Signs have an inner child contour (hole); obstacle is a solid block
        child_idx = hier[0][i][2]
        if child_idx != -1:
            child_area = cv2.contourArea(cnts[child_idx])
            if child_area > 0.15 * area:
                continue

        # Check white strictly INSIDE the contour (prevents road dividers outside from rejecting the obstacle)
        c_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        inner_white = white_global[c_mask > 0]
        if len(inner_white) > 0 and np.mean(inner_white > 0) > 0.15:
            continue

        candidates.append((area, c))

    if not candidates:
        return {"present": False, "area": 0}

    candidates.sort(key=lambda item: item[0], reverse=True)
    largest_area = candidates[0][0]
    return {"present": True, "area": int(round(float(largest_area)))}


def _isolate_red(hsv):
    '''Isolate Red color with hue wrap-around handling.'''
    r1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    r2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    return cv2.bitwise_or(r1, r2)


def _isolate_blue(hsv):
    '''Isolate Blue color for Parking and Hospital signs.'''
    return cv2.inRange(hsv, (100, 90, 50), (135, 255, 255))


def _classify_blue_letter(white_mask):
    '''
    Differentiate between Parking (P) and Hospital (H):
    - Letter "P" has an enclosed loop/hole in the upper half and an empty bottom-right quadrant.
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
    ratio = br / (tr + 1e-5)
    if has_hole or (ratio < 0.72):
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