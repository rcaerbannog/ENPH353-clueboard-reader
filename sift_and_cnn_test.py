import sift_and_cnn as cnn
import cv2 as cv

# SIFT TEST
frame_color = cv.imread('sign.png')
frame_gray = cv.cvtColor(frame_color, cv.COLOR_BGR2GRAY)
warped_frame = cnn.clueboard_img_from_frame(frame_color)
clue_type, clue_value = cnn.clue_type_and_value(warped_frame)
print(clue_type)
print(clue_value)