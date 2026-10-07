'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1A of Niti Vahan (NV) Theme.
*
*  Filename:         ackermann_steering.py
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
# Author List:      Pankaj Amrate , Karan Singh , Krishna Sharma , Prasoon Dhakad
# Functions:        ackermann_wheel_angles
# Global variables: None



####################### IMPORT MODULES #######################
import math
import numpy as np
##############################################################


#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################


def ackermann_wheel_angles(delta):
    '''
    Purpose:
    ---
    Convert a single virtual steering angle into the two real front-wheel
    angles, per the Ackermann geometry.

    Input Arguments:
    ---
    `delta` :   [ float ]
        Steering angle of the virtual centred front wheel, in radians.

    Returns:
    ---
    `left_angle`  : [ float ]
    `right_angle` : [ float ]
        The two real front-wheel steering angles, in radians, using the
        same sign convention as delta.

    REMEMBER:
    ---
    WHEEL_OFFSET changes the effective half-track width inside each wheel's triangle.
    '''
    
    # Calculate effective half-track (distance from vehicle center to kingpin axis)
    effective_half_track = (TRACK_WIDTH / 2.0) - WHEEL_OFFSET
    
    # Using sin() and cos() formulations with atan2 avoids ZeroDivisionError when delta is 0
    # Numerator represents the opposite side of the steering triangle 
    y = WHEELBASE * math.sin(delta)
    
    # Denominators represent the adjacent sides of the left and right steering triangles
    # A positive delta means turning left, so the left wheel is the inner wheel
    x_left = (WHEELBASE * math.cos(delta)) - (effective_half_track * math.sin(delta))
    x_right = (WHEELBASE * math.cos(delta)) + (effective_half_track * math.sin(delta))
    
    # Calculate the final real steering angles
    left_angle = math.atan2(y, x_left)
    right_angle = math.atan2(y, x_right)

    return left_angle, right_angle


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

if __name__ == "__main__":

    test_angles = np.arange(-0.35, 0.35, 0.05)

    for d in test_angles:
        left, right = ackermann_wheel_angles(d)
        print(f"delta={d}  ->  left={left}, right={right}")
