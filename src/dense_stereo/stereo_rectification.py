# Machine Vision
# Assignment 3
# Author: Christopher Mitchell
# Date: 29/03/25

# ========== SECTION 2 ========== #

# ========== Image Rectification ========== #

import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

# ========== Load data ========== #
intrinsics = loadmat("intrinsics.mat")
extrinsics = loadmat("extrinsics.mat")
img1 = cv2.imread("im1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("im2.png", cv2.IMREAD_GRAYSCALE)

K1 = intrinsics["K1"]
K2 = intrinsics["K2"]
C1 = extrinsics["C1"]
C2 = extrinsics["C2"]

R1 = C1[:, :3]
t1 = C1[:, 3].reshape(3, 1)
R2 = C2[:, :3]
t2 = C2[:, 3].reshape(3, 1)

# ========== Rectification Function ========== #
def rectify_images(K1, K2, R1, R2, t1, t2, img1, img2):
    R = R2 @ R1.T
    T = t2 - t1
    h, w = img1.shape

    R1p, R2p, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, None, K2, None, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, None, R1p, P1[:, :3], (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, None, R2p, P2[:, :3], (w, h), cv2.CV_32FC1)

    img1_rect = cv2.remap(img1, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, map2x, map2y, interpolation=cv2.INTER_LINEAR)

    # Optional: crop to valid region of interest
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    img1_rect = img1_rect[y1:y1+h1, x1:x1+w1]
    img2_rect = img2_rect[y2:y2+h2, x2:x2+w2]

    # Outputs
    M1 = R1p
    M2 = R2p
    K1p = P1[:, :3]
    K2p = P2[:, :3]
    t1p = P1[:, 3].reshape(3, 1)
    t2p = P2[:, 3].reshape(3, 1)

    return img1_rect, img2_rect, M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

# ========== Call Function ========== #
img1_rect, img2_rect, M1, M2, K1p, K2p, R1p, R2p, t1p, t2p = rectify_images(K1, K2, R1, R2, t1, t2, img1, img2)

# ========== Print Output Matrices ========== #
np.set_printoptions(precision=4, suppress=True)
print("\n========== Rectification Results ==========\n")
print("M1 (Rectification Matrix 1):\n", M1)
print("\nM2 (Rectification Matrix 2):\n", M2)
print("\nK1p (New Intrinsic Matrix 1):\n", K1p)
print("\nK2p (New Intrinsic Matrix 2):\n", K2p)
print("\nR1p (New Rotation Matrix 1):\n", R1p)
print("\nR2p (New Rotation Matrix 2):\n", R2p)
print("\nt1p (New Translation Vector 1):\n", t1p)
print("\nt2p (New Translation Vector 2):\n", t2p)
print("\n==========================================\n")

# ========== Save Rectified Images ========== #
cv2.imwrite("rectified1_gray.png", img1_rect)
cv2.imwrite("rectified2_gray.png", img2_rect)

# ========== Display Rectified Images ========== #
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img1_rect, cmap="gray")
plt.title("Rectified Image 1")
plt.subplot(1, 2, 2)
plt.imshow(img2_rect, cmap="gray")
plt.title("Rectified Image 2")
plt.tight_layout()
plt.show()
