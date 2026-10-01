# Stereo Vision & 3D Reconstruction

A Python and OpenCV implementation of a **stereo computer vision pipeline** for recovering 3D information from a pair of 2D images.

The project explores the geometry underlying stereo vision, progressing from estimation of the **Fundamental Matrix** and epipolar constraints through camera-pose recovery, triangulation and sparse 3D reconstruction, before implementing stereo rectification, dense correspondence, disparity estimation and depth mapping.

This project was originally developed as part of university Machine Vision coursework and has been reorganised and documented for portfolio presentation.

---

## Project Overview

Stereo vision estimates three-dimensional structure by observing a scene from two different camera viewpoints.

The complete pipeline implemented in this project is:

```text
Stereo Image Pair
       │
       ▼
Known Point Correspondences
       │
       ▼
Eight-Point Algorithm
       │
       ▼
Fundamental Matrix
       │
       ▼
Epipolar Geometry
       │
       ▼
Epipolar Correspondence
       │
       ▼
Essential Matrix
       │
       ▼
Camera Pose Estimation
       │
       ▼
Triangulation
       │
       ▼
Sparse 3D Reconstruction
       │
       ▼
Stereo Rectification
       │
       ▼
Dense Window Matching
       │
       ▼
Disparity Map
       │
       ▼
Depth Estimation
```

The project therefore demonstrates both **sparse geometric reconstruction** and **dense stereo depth estimation**.

---

# Technologies

- Python
- NumPy
- OpenCV
- SciPy
- Matplotlib
- Linear algebra
- Singular Value Decomposition (SVD)
- Epipolar geometry
- Stereo vision
- 3D reconstruction

---

# Repository Structure

```text
stereo-vision-3d-reconstruction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   │
│   ├── sparse_reconstruction/
│   │   ├── fundamental_matrix_and_epipolar_lines.py
│   │   ├── epipolar_correspondence.py
│   │   ├── essential_matrix.py
│   │   ├── triangulation_and_pose_selection.py
│   │   └── stereo_3d_reconstruction.py
│   │
│   └── dense_stereo/
│       ├── stereo_rectification.py
│       ├── dense_disparity.py
│       └── depth_from_disparity.py
│
└── results/
    ├── epipolar_geometry.png
    ├── epipolar_correspondences.png
    ├── sparse_3d_reconstruction.png
    ├── rectified_stereo_pair.png
    ├── disparity_map.png
    └── depth_map.png
```

The original coursework files were organised by assignment section and task number. For portfolio presentation, the scripts have been renamed according to their functionality and divided into **sparse reconstruction** and **dense stereo** stages.

---

# Part 1 — Sparse Stereo Reconstruction

The first half of the project investigates how the geometry between two camera views can be recovered and used to reconstruct points in three-dimensional space.

---

## 1. Fundamental Matrix — Eight-Point Algorithm

The first stage estimates the **Fundamental Matrix**, `F`, from corresponding points in two images.

The Fundamental Matrix describes the projective geometric relationship between two camera views.

For corresponding homogeneous image points:

```text
x₁ ↔ x₂
```

the epipolar constraint is:

```text
x₂ᵀ F x₁ = 0
```

The project implements the **Eight-Point Algorithm** directly.

```text
Point Correspondences
        │
        ▼
Normalise Coordinates
        │
        ▼
Construct Linear System
        │
        ▼
       SVD
        │
        ▼
Estimate F
        │
        ▼
Enforce Rank-2 Constraint
        │
        ▼
Denormalise
        │
        ▼
Fundamental Matrix
```

The implementation constructs the linear system from the point correspondences and solves it using **Singular Value Decomposition (SVD)**.

Because a valid Fundamental Matrix has rank 2, a second SVD is performed and the smallest singular value is set to zero.

The matrix is then transformed back into the original image-coordinate system.

---

## 2. Epipolar Geometry

Once the Fundamental Matrix has been estimated, corresponding points are constrained by **epipolar geometry**.

A point observed in one camera does not need to be searched for across the entire second image.

Instead, its corresponding location must lie along an **epipolar line**.

```text
Image 1

     ● Point
      \
       \
        \  Fundamental Matrix
         \
          ▼

Image 2

────────────────────────  Epipolar Line
```

The project calculates and visualises these epipolar lines for corresponding points in both grayscale and colour images.

Example result:

![Epipolar geometry](results/epipolar_geometry.png)

This provides a visual verification of the estimated two-view geometry.

---

# 3. Epipolar Correspondence

The next stage implements an image correspondence algorithm.

For each reference point in the first image:

```text
Reference Point
      │
      ▼
Compute Epipolar Line
      │
      ▼
Extract Local Image Patch
      │
      ▼
Search Candidate Positions
Along Epipolar Line
      │
      ▼
Compare Image Patches
      │
      ▼
Minimum SSD
      │
      ▼
Corresponding Point
```

A **7 × 7 image window** is extracted around the reference point.

Candidate locations are searched along the corresponding epipolar line in the second image.

Similarity between the reference and candidate patches is measured using **Sum of Squared Differences (SSD)**:

```text
SSD = Σ (I₁ - I₂)²
```

The candidate producing the smallest SSD is selected as the correspondence.

Example:

![Epipolar correspondences](results/epipolar_correspondences.png)

Using the epipolar constraint significantly reduces the correspondence-search problem compared with searching the entire second image.

---

# 4. Essential Matrix

When the camera intrinsic calibration matrices are known, the Fundamental Matrix can be converted into an **Essential Matrix**.

The project computes:

```text
E = K₂ᵀ F K₁
```

where:

```text
F  = Fundamental Matrix
K₁ = Camera 1 Intrinsic Matrix
K₂ = Camera 2 Intrinsic Matrix
E  = Essential Matrix
```

Conceptually:

```text
Fundamental Matrix
        +
Camera Intrinsics
        │
        ▼
 Essential Matrix
        │
        ▼
Relative Camera Geometry
```

Unlike the Fundamental Matrix, which relates image coordinates, the Essential Matrix represents the relationship between the cameras in calibrated coordinates.

---

# 5. Camera Pose Recovery

The Essential Matrix is decomposed using SVD to recover possible relative poses for the second camera.

This produces combinations of:

```text
Rotation R
     +
Translation t
```

There are four possible camera configurations:

```text
             Essential Matrix
                    │
                    ▼
                   SVD
                    │
          ┌─────────┴─────────┐
          │                   │
         R₁                  R₂
      ┌───┴───┐           ┌───┴───┐
      ▼       ▼           ▼       ▼
     +t      -t          +t      -t
```

The project evaluates these candidate camera poses during triangulation.

---

# 6. Triangulation

Once corresponding image points and camera projection matrices are available, their three-dimensional positions can be estimated.

For every correspondence:

```text
Camera 1                       Camera 2
   ●                              ●
    \                            /
     \                          /
      \                        /
       \                      /
        \                    /
         \                  /
          \                /
           ● 3D Point
```

The project constructs a linear triangulation system for each pair of image observations and solves it using SVD.

The resulting homogeneous coordinate is converted into a 3D Cartesian position:

```text
X = [X, Y, Z]
```

---

# 7. Reprojection Error

The reconstructed 3D points are projected back into both camera images.

The difference between the observed image coordinates and reconstructed projections provides a **reprojection error**.

```text
3D Point
   │
   ▼
Camera Projection
   │
   ▼
Predicted 2D Position
   │
   ▼
Compare With
Observed Position
   │
   ▼
Reprojection Error
```

This provides a quantitative method for evaluating reconstruction consistency.

The implementation calculates errors for both camera views and determines a mean reprojection error.

---

# 8. Camera Pose Selection

Because decomposition of the Essential Matrix produces four possible camera poses, the correct configuration must be selected.

Each candidate pose is evaluated using the reconstructed points.

The implementation favours configurations where:

```text
More than 90% of reconstructed points
have positive depth
```

and selects the valid candidate with the lowest reprojection error.

Conceptually:

```text
Four Camera Poses
       │
       ▼
Triangulate Points
       │
       ▼
Positive Depth Test
       │
       ▼
Reprojection Error
       │
       ▼
Select Best Pose
```

This resolves the ambiguity introduced when decomposing the Essential Matrix.

---

# 9. Sparse 3D Reconstruction

With the selected camera pose, the image correspondences are triangulated to form a three-dimensional reconstruction.

The reconstruction pipeline combines:

```text
Fundamental Matrix
        │
        ▼
Epipolar Correspondence
        │
        ▼
Essential Matrix
        │
        ▼
Camera Pose
        │
        ▼
Triangulation
        │
        ▼
3D Point Cloud
```

The resulting 3D points are displayed using Matplotlib.

Example:

![Sparse 3D reconstruction](results/sparse_3d_reconstruction.png)

The recovered camera extrinsic matrices are also saved for use by the dense stereo stage.

---

# Part 2 — Dense Stereo Vision

The second half of the project moves from reconstructing selected points to estimating information across the stereo image pair.

The pipeline is:

```text
Stereo Images
      │
      ▼
Camera Calibration
      │
      ▼
Stereo Rectification
      │
      ▼
Dense Correspondence
      │
      ▼
Disparity
      │
      ▼
Depth
```

---

# 10. Stereo Rectification

Stereo rectification transforms the two camera images so that corresponding points lie along matching horizontal scanlines.

Before rectification:

```text
Image 1                 Image 2

    ●
      \                     ●
       \                  /
        \               /
         Epipolar Geometry
```

After rectification:

```text
Image 1                 Image 2

──────●────────      ──────●────────
```

This greatly simplifies dense correspondence because the search becomes predominantly one-dimensional.

The implementation uses the recovered camera intrinsics and extrinsics to calculate the rectification transformations.

OpenCV is then used to generate remapping matrices and transform both images into their rectified coordinate systems.

Example:

![Rectified stereo pair](results/rectified_stereo_pair.png)

---

# 11. Dense Window Matching

A dense stereo matcher is implemented to calculate disparity across the rectified image pair.

For each pixel in the left image:

```text
Left Image Window
       │
       ▼
Search Along
Horizontal Direction
       │
       ▼
Candidate Windows
       │
       ▼
Calculate SSD
       │
       ▼
Lowest SSD
       │
       ▼
Best Disparity
```

The implementation uses a:

```text
7 × 7 matching window
```

and searches disparities up to:

```text
64 pixels
```

Similarity is again evaluated using **Sum of Squared Differences**.

Rather than relying entirely on a pre-built OpenCV stereo matcher, this stage explicitly implements the window-search process.

---

# 12. Disparity Map

The displacement between corresponding pixels in the left and right images is known as **disparity**.

```text
Left Image                Right Image

       ●                         ●
       │<------ disparity ------>│
```

A disparity value is estimated for each pixel for which a correspondence can be determined.

The resulting values form a **disparity map**.

Example:

![Disparity map](results/disparity_map.png)

Regions with different disparities correspond to scene points at different distances from the stereo cameras.

---

# 13. Depth Estimation

Disparity can be converted into an estimate of depth using the stereo-camera geometry.

The project uses:

```text
        baseline × focal length
depth = ───────────────────────
                disparity
```

where:

```text
baseline     = distance between camera optical centres
focal length = camera focal length
disparity    = horizontal displacement between corresponding pixels
```

The camera centres are calculated from the rotation and translation components of the camera extrinsics.

```text
Camera 1 ●──────────────● Camera 2
         <---baseline--->
             \      /
              \    /
               \  /
                ●
             3D Point
```

The result is a dense depth representation of the observed scene.

Example:

![Depth map](results/depth_map.png)

---

# Sparse vs Dense Reconstruction

The project demonstrates two complementary stereo-vision approaches.

| Sparse Reconstruction | Dense Stereo |
|---|---|
| Operates on selected image points | Operates across the image |
| Uses epipolar correspondence | Uses dense window matching |
| Recovers individual 3D points | Estimates disparity per pixel |
| Produces a 3D point reconstruction | Produces disparity/depth maps |
| Useful for geometric reconstruction | Useful for scene-depth perception |

Together, they demonstrate how the same stereo-camera geometry can support different forms of 3D perception.

---

# Mathematical Concepts Demonstrated

The project makes extensive use of linear algebra and projective geometry, including:

### Singular Value Decomposition

Used for:

- Fundamental Matrix estimation
- Rank-2 enforcement
- Essential Matrix decomposition
- Linear triangulation

### Epipolar Geometry

Used to constrain correspondence between two camera views.

### Camera Projection

The calibrated camera matrices relate reconstructed 3D points to image coordinates.

### Triangulation

Used to recover a 3D point from observations in two different camera views.

### Reprojection Error

Used to evaluate the consistency of reconstructed points with the original observations.

### Disparity Geometry

Used to recover scene depth from a rectified stereo-camera pair.

---

# Installation

Install the Python dependencies using:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
numpy
opencv-python
scipy
matplotlib
```

---

# Input Data

The original coursework pipeline uses a number of supplied calibration, correspondence and image files, including:

```text
im1.png
im2.png
some_corresp.mat
intrinsics.mat
temple_coords.mat
fundamental.mat
```

These contain the stereo image pair, known point correspondences and camera calibration information used during the experiments.

Depending on licensing or coursework-distribution restrictions, these original supplied files may not be included in this portfolio repository.

---

# Generated Data

During execution, the pipeline can generate intermediate data including:

```text
extrinsics.mat
rectified1_gray.png
rectified2_gray.png
disparity_map.npy
disparity_map.png
```

These represent recovered camera geometry and intermediate outputs used by later stages of the pipeline.

---

# Running the Project

The scripts represent successive stages of the stereo-vision pipeline rather than a single standalone application.

For example, estimation of the Fundamental Matrix and visualisation of epipolar geometry can be performed with:

```bash
python src/sparse_reconstruction/fundamental_matrix_and_epipolar_lines.py
```

The complete sparse reconstruction is implemented in:

```bash
python src/sparse_reconstruction/stereo_3d_reconstruction.py
```

Stereo rectification can then be performed using:

```bash
python src/dense_stereo/stereo_rectification.py
```

followed by disparity estimation:

```bash
python src/dense_stereo/dense_disparity.py
```

and finally depth estimation:

```bash
python src/dense_stereo/depth_from_disparity.py
```

Some stages depend on intermediate files produced by earlier stages or on the original coursework dataset.

---

# Implementation Note — Disparity Representation

During portfolio review, the original disparity implementation was updated to preserve the **true pixel-disparity values** separately from the normalised image used for visualisation.

The original coursework implementation normalised the calculated disparities into an 8-bit `0–255` image representation before saving them.

This was suitable for visualisation but meant that a subsequent depth calculation could inadvertently interpret display-scaled values as physical pixel disparities.

The portfolio version therefore maintains:

```text
True Disparity
      │
      ├──────────────► disparity_map.npy
      │                 Used for depth calculation
      │
      ▼
Normalisation
      │
      ▼
0–255 Visualisation
      │
      └──────────────► disparity_map.png
```

This ensures that the depth relationship:

```text
depth = baseline × focal length / disparity
```

uses disparity measured in pixels rather than display intensity.

---

# Concepts & Skills Demonstrated

This project provided practical experience with:

### Computer Vision

- Stereo vision
- Epipolar geometry
- Image correspondence
- Stereo rectification
- Disparity estimation
- Depth estimation
- 3D reconstruction

### Geometry & Mathematics

- Fundamental Matrix
- Essential Matrix
- Camera intrinsic matrices
- Camera extrinsic matrices
- Rotation and translation
- Homogeneous coordinates
- Linear triangulation
- Singular Value Decomposition
- Reprojection error

### Image Processing

- Grayscale image processing
- Image patches
- Window-based matching
- Sum of Squared Differences
- Image remapping
- Dense pixel correspondence

### Python Development

- NumPy matrix operations
- OpenCV
- SciPy `.mat` file processing
- Matplotlib
- Modular algorithm development
- Numerical computing

---

# Original Coursework

This project originated from university Machine Vision coursework.

The original submission was divided into individual assignment tasks covering progressively more advanced stereo-vision concepts.

For portfolio presentation:

- Scripts were renamed according to their functionality.
- Sparse and dense stereo stages were separated into logical directories.
- Documentation was added around the complete pipeline.
- Outputs were organised into a dedicated `results` directory.
- A disparity representation issue identified during portfolio review was corrected so that numerical disparity and visualisation data remain separate.

The project remains representative of the algorithms and approach developed during the original coursework.

---

# Known Limitations

This project was developed as an educational implementation of stereo-vision algorithms rather than a production stereo-perception system.

Several areas could be developed further.

### Correspondence Matching

The sparse and dense matchers use relatively simple SSD-based image-patch comparisons.

More advanced implementations could investigate:

- Normalised cross-correlation
- Census transform
- Feature-based matching
- Left-right consistency checks
- Subpixel disparity refinement

### Dense Stereo Performance

The dense matcher explicitly loops over image pixels and candidate disparities.

This is useful for demonstrating the underlying algorithm but is computationally expensive.

A production implementation could investigate:

- Vectorisation
- Parallel processing
- GPU acceleration
- OpenCV StereoBM
- OpenCV StereoSGBM

### Occlusions

Not every point visible in one camera is necessarily visible in the other.

A more advanced stereo implementation could explicitly detect and reject occluded regions.

### Confidence Estimation

Future versions could calculate confidence values based on matching quality rather than accepting only the minimum SSD candidate.

---

# Potential Extensions

A future version of the project could extend the pipeline to:

```text
Stereo Cameras
      │
      ▼
Rectification
      │
      ▼
Dense Disparity
      │
      ▼
Depth
      │
      ▼
3D Point Cloud
      │
      ▼
Obstacle Detection
      │
      ▼
Environment Mapping
```

Possible improvements include:

- Dense 3D point-cloud generation
- Open3D visualisation
- StereoSGBM comparison
- Real-time stereo-camera input
- GPU-accelerated correspondence
- Depth filtering
- Point-cloud filtering
- Obstacle detection
- ROS / ROS2 integration

---

# Robotics Context

Stereo vision provides a method for estimating the three-dimensional structure of an environment using passive cameras.

A robotic perception system could use the same principles within a pipeline such as:

```text
Stereo Cameras
      │
      ▼
Image Acquisition
      │
      ▼
Stereo Calibration
      │
      ▼
Rectification
      │
      ▼
Correspondence
      │
      ▼
Disparity
      │
      ▼
Depth
      │
      ▼
3D Environment Representation
      │
      ▼
Perception / Navigation
      │
      ▼
Robot Decision Making
```

The project therefore provides practical experience with mathematical concepts that underpin **robot perception, autonomous navigation and 3D scene understanding**.

---

# Portfolio Context

This project represents the progression from conventional 2D machine vision into **multi-view geometry and three-dimensional perception**.

```text
2D Image Processing
        │
        ▼
Feature Correspondence
        │
        ▼
Epipolar Geometry
        │
        ▼
Camera Geometry
        │
        ▼
Triangulation
        │
        ▼
3D Reconstruction
        │
        ▼
Dense Stereo
        │
        ▼
Depth Perception
        │
        ▼
Robotic Perception
```

Rather than treating stereo reconstruction as a black-box operation, the project implements many of the underlying mathematical stages directly, providing practical experience with the geometry used to recover three-dimensional information from multiple camera views.
