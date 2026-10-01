# Machine Vision
# Assignment 3
# Author: Christopher Mitchell
# Date: 29/03/25

# ========== SECTION 2 ========== #

import numpy as np
import cv2 as cv
from scipy.io import loadmat
import matplotlib.pyplot as plt

def depth_map_from_disparity(dispM, K1, K2, R1, R2, t1, t2):
    """
    Compute depth map from disparity using the formula: depth = (baseline * focal_length) / disparity.
    
    Args:
        dispM: Disparity map (2D array), 0 indicates invalid pixels.
        K1, K2: Intrinsic matrices (3x3) for left/right cameras.
        R1, R2: Rotation matrices (3x3) for left/right cameras.
        t1, t2: Translation vectors (3x1) for left/right cameras.
    
    Returns:
        depthM: Depth map (same shape as dispM), with 0 for invalid disparities.
    """
    # --- Compute baseline (distance between optical centers) ---
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    baseline = np.linalg.norm(c1 - c2)
    
    # --- Get focal length from K1 (assuming fx = fy) ---
    focal_length = K1[0, 0]
    
    # --- Initialize depth map ---
    depthM = np.zeros_like(dispM, dtype=np.float32)
    
    # --- Calculate depth only for valid disparities (dispM > 0) ---
    valid_mask = dispM > 0
    depthM[valid_mask] = (baseline * focal_length) / dispM[valid_mask]
    
    return depthM


# ====== Example Usage ======
if __name__ == "__main__":
    dispM = np.load("disparity_map.npy")  # Load disparity map
    intrinsics = loadmat("intrinsics.mat")
    extrinsics = loadmat("extrinsics.mat")
    
    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]
    C1 = extrinsics["C1"]
    C2 = extrinsics["C2"]
    
    R1, t1 = C1[:, :3], C1[:, 3].reshape(3, 1)
    R2, t2 = C2[:, :3], C2[:, 3].reshape(3, 1)
    
    depthM = depth_map_from_disparity(dispM, K1, K2, R1, R2, t1, t2)
    
    # ====== Print Depth Map Statistics ======
    valid_depths = depthM[depthM > 0]
    print("\n========= Depth Map Stats =========")
    print(f"Valid pixel count  : {valid_depths.size}")
    print(f"Min depth (non-zero): {np.min(valid_depths):.2f}")
    print(f"Max depth           : {np.max(valid_depths):.2f}")
    print(f"Mean depth          : {np.mean(valid_depths):.2f}")
    print("===================================\n")
    
    # ====== Visualize ======
    plt.imshow(depthM, cmap="inferno")
    plt.colorbar(label="Depth")
    plt.title("Depth Map")
    plt.show()
