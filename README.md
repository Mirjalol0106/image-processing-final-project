# image-processing-final-project

# Panorama Image Stitching — Final Project

A web application that combines multiple overlapping photos into 
a seamless panoramic image using computer vision algorithms.

## How it works
1. **SIFT** — detects feature keypoints in each image
2. **FLANN** — matches keypoints between images
3. **RANSAC** — estimates homography to align images
4. **Laplacian Pyramid Blending** — merges images seamlessly

## How to run

pip install flask opencv-python numpy
python3 app.py

Then open http://127.0.0.1:8080 in your browser.

## Technologies
- Python 3
- OpenCV 4
- Flask
- NumPy
