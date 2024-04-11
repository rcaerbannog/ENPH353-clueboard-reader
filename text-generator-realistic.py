import cv2 as cv
import numpy as np
import os
import clueboards

from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

LETTER_IMG_DIM_X = 64
LETTER_IMG_DIM_Y = 80
LETTER_IMG_SHAPE = (LETTER_IMG_DIM_Y, LETTER_IMG_DIM_X)

THRESHOLD_MIN = 128
THRESHOLD_MAX = 255
letter_thresh_minHSV = clueboards.letter_thresh_minHSV
letter_thresh_maxHSV = clueboards.letter_thresh_maxHSV

# MAX_SHIFT = 3
FONT_SIZES = [80, 85, 90, 95, 100]
FONT_SIZE_LOCATIONS = [(11, -3), (11, -5), (10, -7), (9, -9), (7, -12)]

def letter_img_from(letter, font_size):
    blank_letter_color = Image.fromarray(np.full((200, 200, 3), 200, dtype=np.uint8)) #dtype required
    draw = ImageDraw.Draw(blank_letter_color, mode='RGB')
    monospace = ImageFont.truetype("train/UbuntuMono-R.ttf", font_size)
    draw.text((50, 50), letter, fill=(255,0,0), font=monospace) # When read as BGR by cv2, will become blue
    letter_img = np.array(blank_letter_color)
    return letter_img

def mangled(letter_img):
    letter_img_new = cv.medianBlur(letter_img, 11)
    cv.GaussianBlur(letter_img_new, (21,21), sigmaX=0, dst=letter_img_new)
    return letter_img_new

def letter_mask_from(img): # BGR or GRAY
    if len(img.shape) == 2:
        img_mask = cv.inRange(img, THRESHOLD_MIN, THRESHOLD_MAX)
    else:
        img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_mask = cv.inRange(img_HSV, clueboards.letter_thresh_minHSV, clueboards.letter_thresh_maxHSV)
    # contour detection
    contours, hierarchy = cv.findContours(img_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = tuple(sorted(contours, key = lambda x: cv.contourArea(x), reverse = True)) # largest to smallest contours
    # Take the largest contour as the letter
    contour = contours[0]
    x, y, w, h = cv.boundingRect(contour)
        
    subimg = np.copy(img_mask[y:y+h, x:x+w])
    letter_mask = np.zeros(LETTER_IMG_SHAPE, dtype=np.uint8)
    start_x = (LETTER_IMG_DIM_X - w) // 2
    start_y = (LETTER_IMG_DIM_Y - h) // 2
    letter_mask[start_y : start_y + h, start_x : start_x + w] = subimg
    return letter_mask

# Clear the 'train-letters' directory before running.
try:
    os.mkdir('train/train-letters/')
except(OSError):
    pass

for i in range(26):
    # For each letter, make 4*2*2 = 16 images: four font sizes x mangled/not x eroded/not
    i = 0
    letter = chr(ord('A') + i)
    outpath = f'train/train-letters/{letter}'
    try:
        os.mkdir(outpath)
    except(OSError):
        pass
    for font_size in FONT_SIZES: # 4
        letter_img_base = letter_img_from(letter, font_size)
        letter_img_mangled = mangled(letter_img_base)

        letter_imgs = [letter_img_base, letter_img_mangled]
        for letter_img in letter_imgs: # 2
            # mask -> eroded or not
            letter_mask_base = letter_mask_from(letter_img)
            letter_mask_eroded = letter_mask_from(letter_img)
            cv.imwrite(f"train/train_letters/{letter}/{letter}_{i}.png", letter_mask_base)
            i += 1
            cv.imwrite(f"train/train_letters/{letter}/{letter}_{i}.png", letter_mask_eroded)
            i += 1

