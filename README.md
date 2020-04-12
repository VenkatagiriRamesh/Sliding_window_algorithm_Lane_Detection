# Lane-Detection---Curve-Estimation-Method

## The following the steps of image processing and analysis involved in 'detection.py',
1. Compute the camera calibration matrix and distortion coefficients using a set of chessboard images taken from Picamera.

![](/images/5.jpg)

2. Apply a distortion correction to raw images.

![](/images/6.jpg)

3. Use binary transforms., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary using sliding window algorithm.

![](/images/1.png)

6. Determine the curvature of the lane and vehicle position with respect to center.
7. Project the detected lane boundaries back onto the original image.

![](/images/2.png)

8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle offset position.

![](/images/3.png)

## Required Python Libraries
1. cv2 (opencv)
2. numpy
3. matplotlib
4. glob
5. pickle

## Code Execution 
`cd Sliding-window-algorithm---Lane-Detection` 

`python lane_detection.py` 
