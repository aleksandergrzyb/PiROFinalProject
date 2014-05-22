PiROFinalProject
================

This is final project of Image Processing and Recognition course.

The main topic of final project is to create application that would detect all occurrences of given object on given photo.

## Working example

**Object to be found:**

![Detected newspapers on sample image](http://cl.ly/image/3S472i310O2m/Screen%20Shot%202014-05-22%20at%2021.52.43%20.png)

**Detected newspapers on a sample image:**

![Detected newspapers on sample image](http://f.cl.ly/items/0G2F0F0B3G130S210U1C/Screen%20Shot%202014-05-22%20at%2021.52.16%20.png)

## Method

My algorithm is based on [Features2D + Homography][1]. The main difference is that my algorithm is capable of detecting multiple occurrences of given object. I implemented this using 'Sliding Window' mechanism. I am moving window with fixed size around a scene image. In every position of this window I am searching for given object. Sometimes algorithm detects bad frames for object. I reject this error by checking length ratio of every side of query image (image with object that we are looking for) and scene image (image where we are looking for object). If ratio is not almost the same for every side of object I reject this frame. Frames that pass this test are drawn on scene image.

[1]: http://docs.opencv.org/doc/tutorials/features2d/feature_homography/feature_homography.html "Features2D + Homography"
