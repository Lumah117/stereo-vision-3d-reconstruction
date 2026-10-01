# Machine Vision
# Assignment 3
# Author: Christopher Mitchell
# Date: 25/03/25

# ========== SECTION 1 ========== #

# ========== Part 1: Implement the Eight-Point Algorithm ========== #

# Code for imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import loadmat


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

# ========== Draw Epipolar Lines ==========
def draw_epipolar_lines(img1, img2, F, pts1, pts2):
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F).reshape(-1,3)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F).reshape(-1,3)

    def draw_lines(img, lines, pts):
        img = img.copy()
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        for r, pt in zip(lines, pts):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [w, -(r[2] + r[0]*w)/r[1]])
            img = cv.line(img, (x0,y0), (x1,y1), color, 1)
            img = cv.circle(img, tuple(pt.astype(int)), 5, color, -1)
        return img

    img1_with_lines = draw_lines(img1, lines1, pts1)
    img2_with_lines = draw_lines(img2, lines2, pts2)

    return img1_with_lines, img2_with_lines

# ========== Load Data ==========
data = loadmat('some_corresp.mat')
pts1 = data['pts1']
pts2 = data['pts2']

img1_color = cv.imread('im1.png')
img2_color = cv.imread('im2.png')

img1_gray = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

M = max(img1_color.shape[0], img1_color.shape[1])

# ========== Estimate Fundamental Matrix ==========
F = eight_point(pts1, pts2, M)
print(" Fundamental Matrix (F):\n", F)

# ========== Epipolar Lines ==========
pts1 = pts1.astype(np.float32)
pts2 = pts2.astype(np.float32)

gray1_with_lines, gray2_with_lines = draw_epipolar_lines(img1_gray, img2_gray, F, pts1, pts2)
color1_with_lines, color2_with_lines = draw_epipolar_lines(img1_color, img2_color, F, pts1, pts2)

# ========== Plotting ==========
# Show original color images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(cv.cvtColor(img1_color, cv.COLOR_BGR2RGB)), plt.title("Original Image 1")
plt.subplot(1,2,2), plt.imshow(cv.cvtColor(img2_color, cv.COLOR_BGR2RGB)), plt.title("Original Image 2")
plt.tight_layout(), plt.show()

# Show grayscale with epipolar lines
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(cv.cvtColor(gray1_with_lines, cv.COLOR_BGR2RGB)), plt.title("Grayscale with Epilines 1")
plt.subplot(1,2,2), plt.imshow(cv.cvtColor(gray2_with_lines, cv.COLOR_BGR2RGB)), plt.title("Grayscale with Epilines 2")
plt.tight_layout(), plt.show()

# Show color with epipolar lines
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(cv.cvtColor(color1_with_lines, cv.COLOR_BGR2RGB)), plt.title("Color with Epilines 1")
plt.subplot(1,2,2), plt.imshow(cv.cvtColor(color2_with_lines, cv.COLOR_BGR2RGB)), plt.title("Color with Epilines 2")
plt.tight_layout(), plt.show()
