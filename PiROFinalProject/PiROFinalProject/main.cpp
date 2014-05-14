//
//  main.cpp
//  PiROFinalProject
//
//  Created by Aleksander Grzyb on 14/05/14.
//  Copyright (c) 2014 Aleksander Grzyb. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char * argv[])
{
    Mat queryImage = imread("/Users/AleksanderGrzyb/Desktop/PiROFinalProject/Sample Images/Newspapers/object.JPG");
    int minHessian = 400;
    SurfDescriptorExtractor detector(minHessian);
    vector<KeyPoint> keyPoints;
    detector.detect(queryImage, keyPoints);
    Mat keyPointsImage;
    drawKeypoints(queryImage, keyPoints, keyPointsImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("Keypoints", keyPointsImage);
    waitKey(0);
    return 0;
}


