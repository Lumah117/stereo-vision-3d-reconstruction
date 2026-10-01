# Machine Vision
# Assignment 3
# Author: Christopher Mitchell
# Date: 25/03/25

# ========== SECTION 1 ========== #

# ========== Full 3D Reconstruction Script ========== #

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from mpl_toolkits.mplot3d import Axes3D

# ========== Load Data ========== #
intrinsics = loadmat("intrinsics.mat")
corresp = loadmat("some_corresp.mat")
temple = loadmat("temple_coords.mat")

K1 = intrinsics["K1"]
K2 = intrinsics["K2"]
pts1_init = corresp["pts1"]
pts2_init = corresp["pts2"]
pts1 = temple["pts1"]  # 2D points in image 1 to match

img1 = cv.imread("im1.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("im2.png", cv.IMREAD_GRAYSCALE)
M = max(img1.shape)  # For normalization

# ========== 1. Eight-Point Algorithm ========== #
def eight_point(pts1, pts2, M):
    pts1_norm = pts1 / M
    pts2_norm = pts2 / M
    N = pts1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    Uf, Sf, Vtf = np.linalg.svd(F)
    Sf[-1] = 0
    F = Uf @ np.diag(Sf) @ Vtf
    T = np.diag([1/M, 1/M, 1])
    F = T.T @ F @ T
    return F

# ========== 2. Essential Matrix ========== #
def compute_essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

# ========== 3. Get Camera 2 Poses ========== #
def camera2(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [
        np.hstack((R1,  t.reshape(3, 1))),
        np.hstack((R1, -t.reshape(3, 1))),
        np.hstack((R2,  t.reshape(3, 1))),
        np.hstack((R2, -t.reshape(3, 1))),
    ]

# ========== 4. Subpixel Epipolar Correspondence ========== #
def epipolar_correspondence(im1, im2, F, x1, y1):
    window_size = 5
    search_range = 10
    half_win = window_size // 2
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    p1 = np.array([x1, y1, 1.0])
    line = F @ p1
    a, b, c = line
    best_ssd = np.inf
    best_pt = (x1, y1)

    if y1 - half_win < 0 or y1 + half_win >= im1.shape[0] or x1 - half_win < 0 or x1 + half_win >= im1.shape[1]:
        return best_pt

    template = cv.getRectSubPix(im1, (window_size, window_size), (x1, y1))
    template = (template - np.mean(template)) / (np.std(template) + 1e-5)

    for x_can in np.linspace(x1 - search_range, x1 + search_range, 40):
        if abs(b) > 1e-12:
            y_can = -(a * x_can + c) / b
        else:
            continue
        if y_can - half_win >= 0 and y_can + half_win < im2.shape[0] and x_can - half_win >= 0 and x_can + half_win < im2.shape[1]:
            patch = cv.getRectSubPix(im2, (window_size, window_size), (x_can, y_can))
            patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)
            ssd = np.sum((template - patch) ** 2)
            if ssd < best_ssd:
                best_ssd = ssd
                best_pt = (x_can, y_can)

    return best_pt

# ========== 5. Triangulation ========== #
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    pts3d = []
    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.array([
            y1 * P1[2] - P1[1],
            P1[0] - x1 * P1[2],
            y2 * P2[2] - P2[1],
            P2[0] - x2 * P2[2]
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]
        pts3d.append(X[:3])

    pts3d = np.array(pts3d)
    proj1 = P1 @ np.hstack((pts3d, np.ones((N, 1)))).T
    proj2 = P2 @ np.hstack((pts3d, np.ones((N, 1)))).T
    proj1 /= proj1[2]
    proj2 /= proj2[2]

    errors1 = np.sqrt((pts1[:, 0] - proj1[0])**2 + (pts1[:, 1] - proj1[1])**2)
    errors2 = np.sqrt((pts2[:, 0] - proj2[0])**2 + (pts2[:, 1] - proj2[1])**2)
    per_point_error = (errors1 + errors2) / 2
    err = np.mean(per_point_error)
    return pts3d, err

# ========== Step 1: Compute F ========== #
F = eight_point(pts1_init, pts2_init, M)

# ========== Step 2: Match pts2 via epipolar correspondence ========== #
pts2 = []
for i in range(pts1.shape[0]):
    x1, y1 = pts1[i]
    x2, y2 = epipolar_correspondence(img1, img2, F, x1, y1)
    pts2.append([x2, y2])
pts2 = np.array(pts2)

# ========== Step 3: Camera matrices ========== #
E = compute_essential_matrix(F, K1, K2)
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2_candidates = [K2 @ P for P in camera2(E)]

# ========== Step 4: Choose correct P2 ========== #
best_pts3d, best_err, best_P2 = None, float("inf"), None
for P2 in P2_candidates:
    pts3d, err = triangulate(P1, pts1, P2, pts2)
    if np.mean(pts3d[:, 2] > 0) > 0.9 and err < best_err:
        best_pts3d = pts3d
        best_err = err
        best_P2 = P2

# ========== Step 5: Save extrinsics ========== #
C1 = np.hstack((np.eye(3), np.zeros((3, 1))))
C2 = np.linalg.inv(K2) @ best_P2
savemat("extrinsics.mat", {"C1": C1, "C2": C2})

# ========== Final Stats ========== #
z_vals = best_pts3d[:, 2]
print("Final Reconstruction Stats:")
print(f"Z Range: min={z_vals.min():.2f}, max={z_vals.max():.2f}, mean={z_vals.mean():.2f}")
print(f"Points behind camera: {np.sum(z_vals < 0)} / {len(z_vals)}")
print(f"Final Reprojection Error: {best_err:.4f}")
print(f"Number of 3D Points: {best_pts3d.shape[0]}")

# ========== 3D Plot ========== #
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], s=2, c='teal')
ax.set_title("Full 3D Reconstruction of Temple")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=30, azim=135)
plt.tight_layout()
plt.show()
