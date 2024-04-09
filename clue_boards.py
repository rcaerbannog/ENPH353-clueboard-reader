#!/usr/bin/env python3
import cv2 as cv
import numpy as np

def is_good_clueboard_contour(contour, in_frame):
    MIN_HEIGHT = 130
    frame_dimY = in_frame.shape[0]
    frame_dimX = in_frame.shape[1]
    x, y, w, h = cv.boundingRect(contour)
    # Failure conditions: height too small, contour touches edge of screen (indicates partly out of frame)
    return not (h < MIN_HEIGHT or x == 0 or y == 0 or x+w >= frame_dimX-1 or y+h >= frame_dimY-1)

## Call this
def clueboard_img_from_frame(frame):
    DIM_Y = 400
    DIM_X = 600
    
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #upper and lower blue bounds
    upper_blue = np.array([150, 255, 255])
    lower_blue = np.array([120, 50, 50])

    #create mask
    mask = cv.inRange(frame_hsv, lower_blue, upper_blue)
    #find contours
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda x : cv.contourArea(x) >= 2500, contours)
    sorted_contours = tuple(sorted(contours, key = lambda x: cv.contourArea(x), reverse = True)) # largest to smallest contours
    
    # At the very least, we should detect an inner and outer contour of the blue border.
    # If part of the clueboard is out-of-frame, we can tell by the largest contour touching the edge. (Check bounding box)
    if len(sorted_contours) >= 2:
        clueboard_contour = sorted_contours[1] # Hopefully the second contour is the interior one
        if is_good_clueboard_contour(clueboard_contour, mask):
            # Get bounding quadrilateral
            perim = cv.arcLength(clueboard_contour, closed=True)
            quadrilateral = cv.approxPolyDP(clueboard_contour, epsilon=0.02*perim, closed=True)
            if len(quadrilateral) == 4:
                quad_pts = np.array([item for sublist in quadrilateral for item in sublist], np.float32)
                # Rotate the quadrilateral to order: top-left, top-right, bottom-right, bottom-left
                s = np.array(sorted(quad_pts, key=lambda i: i[1], reverse=False)) # Sort by y-coord. Top / Bot sep
                top_left, top_right = (0, 1) if s[0][0] < s[1][0] else (1, 0)
                bot_left, bot_right = (2, 3) if s[2][0] < s[3][0] else (3, 2)
                quad_pts_std = np.array([s[top_left], s[top_right], s[bot_right], s[bot_left]], dtype=np.float32)
                # the points to transform to (clueboard template image 600x400)
                h = np.array([ [0,0],[DIM_X-1,0],[DIM_X-1,DIM_Y-1],[0,DIM_Y-1] ],np.float32)
                transform = cv.getPerspectiveTransform(quad_pts_std, h)
                clueboard_img = cv.warpPerspective(frame, transform, (DIM_X, DIM_Y))
                return True, clueboard_img # Warped clueboard image (masked), ready for processing
        
    return False, frame # Did not detect clue
