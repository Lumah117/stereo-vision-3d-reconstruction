# Machine Vision
# Assignment 3
# Author: Christopher Mitchell
# Date: 29/03/25

# ========== SECTION 2 ========== #

# ========== Part 2: Dense Window Matching for Disparity ========== #

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ========== Load rectified images ========== #
im1 = cv.imread('rectified1_gray.png', cv.IMREAD_GRAYSCALE)
im2 = cv.imread('rectified2_gray.png', cv.IMREAD_GRAYSCALE)

if im1 is None or im2 is None:
    raise FileNotFoundError("Could not load rectified images. Check paths!")

# Resize im2 to match im1's dimensions (if they're slightly off)
if im1.shape != im2.shape:
    im2 = cv.resize(im2, (im1.shape[1], im1.shape[0]))

# ========== Disparity Function ========== #
def compute_disparity_map(im1, im2, max_disp=64, win_size=7):
    assert im1.shape == im2.shape, "Images must be the same size!"
    h, w = im1.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    half_win = win_size // 2

    # Pad both images just for windowing
    padded_im1 = cv.copyMakeBorder(im1, half_win, half_win, half_win, half_win, cv.BORDER_REFLECT)
    padded_im2 = cv.copyMakeBorder(im2, half_win, half_win, half_win, half_win + max_disp, cv.BORDER_REFLECT)

    for y in range(h):
        for x in range(w):
            min_ssd = float('inf')
            best_offset = 0

            # Define window in im1 (left)
            x1 = x + half_win
            y1 = y + half_win
            window1 = padded_im1[y1 - half_win:y1 + half_win + 1, x1 - half_win:x1 + half_win + 1]

            for disp in range(max_disp + 1):
                x2 = x1 - disp
                if x2 - half_win < 0:
                    continue  # out of bounds on the right image

                window2 = padded_im2[y1 - half_win:y1 + half_win + 1, x2 - half_win:x2 + half_win + 1]

                ssd = np.sum((window1.astype(np.float32) - window2.astype(np.float32)) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = disp

            disparity_map[y, x] = best_offset

    # Normalize to 0–255
    disparity_map = (disparity_map / max_disp) * 255
    return disparity_map.astype(np.uint8)


# ========== Compute disparity ========== #
disparity = compute_disparity_map(im1, im2, max_disp=64, win_size=7)

# ========== Save results ========== #
np.save("disparity_map.npy", disparity)
cv.imwrite("disparity_map.png", disparity)

# ========== Plot ========== #
plt.figure(figsize=(10, 6))
plt.imshow(disparity, cmap='plasma')
plt.colorbar(label='Disparity (pixels)')
plt.title('Disparity Map')
plt.axis('off')
plt.tight_layout()
plt.show()
