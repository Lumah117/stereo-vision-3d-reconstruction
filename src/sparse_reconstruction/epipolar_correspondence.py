# Machine Vision
# Assignment 3
# Author: Christopher Mitchell
# Date: 25/03/25

# ========== SECTION 1 ========== #

# ========== Part 2: Find Epipolar Correspondence ========== #

# Code for imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ========== Eight-Point Algorithm ==========
# Function for Eight-Point Algorithm
def eight_point(pts1, pts2, M):
    # Normalize points with scale and translation
    pts1_norm = pts1/M
    pts2_norm = pts2/M

    # Construct matrix A
    N = pts1_norm.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x1 * x2, x1 * y2, x1,
                y1 * x2, y1 * y2, y1,
                x2, y2, 1]

    # Solve Af = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint
    Uf, Sf, Vtf = np.linalg.svd(F_norm)
    Sf[-1] = 0
    F_norm_rank2 = Uf @ np.diag(Sf) @ Vtf
    T = np.diag([1/M, 1/M, 1])

    # Denormalize
    F = T.T @ F_norm_rank2 @ T
    return F

# ========== Epipolar Correspondence Matching ==========
def epipolar_correspondences(im1, im2, F, pts1):
    window_size = 7
    search_range = 15

    if len(im1.shape) == 3:
            im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

    if len(im2.shape) ==3:
            im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    half_win = window_size //2
    pts2 = []

    for (x1, y1) in pts1:
          p1 = np.array([x1, y1, 1.0])
          line = F@p1
          a, b, c = line
          best_ssd = np.inf
          best_pt = (x1, y1)

          x1_int, y1_int = int(round(x1)), int(round(y1))
          if (y1_int - half_win < 0 or y1_int + half_win >= im1.shape[0] or x1_int - half_win < 0 or x1_int + half_win >= im1.shape[1]):
                pts2.append(best_pt)
                continue
          
          template = im1[y1_int - half_win:y1_int + half_win + 1, x1_int-half_win:x1_int + half_win +1]

          for x_can in np.arange(x1-search_range, x1 + search_range + 1):
                if abs(b) > 1e-12:
                      y_can = -(a*x_can + c) / b
                else:
                      continue
                x_can_int = int(round(x_can))
                y_can_int = int(round(y_can))

                if (y_can_int - half_win >= 0 and y_can_int + half_win < im2.shape[0] and x_can_int - half_win >= 0 and x_can_int + half_win < im2.shape[1]):
                      patch = im2[y_can_int - half_win: y_can_int + half_win + 1, x_can_int - half_win: x_can_int + half_win + 1]
                      ssd = np.sum((template - patch)**2)
                      if ssd < best_ssd:
                            best_ssd = ssd
                            best_pt = (x_can_int, y_can_int)
          pts2.append(best_pt)
    return np.array(pts2)      
                

# ========== Load Data ==========
data = loadmat("some_corresp.mat")
pts1 = data["pts1"]
pts2 = data["pts2"]

img1_color = cv2.imread("im1.png")
img2_color = cv2.imread("im2.png")
img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# ========== Estimate Fundamental Matrix ==========
M = max(img1_color.shape[0], img1_color.shape[1])
F = eight_point(pts1, pts2, M)

# ========== Use a subset of pts1 to find matches ==========
pts1_subset = pts1#[::10]  # You can also try [::5] or [::25] for testing
pts2_matched = epipolar_correspondences(img1_gray, img2_gray, F, pts1_subset)

# ========== Visualize Matches ==========
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
plt.scatter(pts1_subset[:, 0], pts1_subset[:, 1], c='r')
plt.title("Image 1: Reference Points")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
plt.scatter(pts2_matched[:, 0], pts2_matched[:, 1], c='g')
plt.title("Image 2: Matched Points (Epipolar)")

plt.tight_layout()
plt.show()
