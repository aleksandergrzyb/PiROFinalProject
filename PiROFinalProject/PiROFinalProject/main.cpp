//
//  main.cpp
//  PiROFinalProject
//
//  Created by Aleksander Grzyb on 14/05/14.
//  Copyright (c) 2014 Aleksander Grzyb. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void sortMatchesToFindGoodOnes(vector<DMatch>& allMatches, int queryDesctiptorSize, vector<DMatch>& goodMatches)
{
    // Calculation of maximum and minimum distances between keypoints
    double maxDistance = 0;
    double minDistance = 100;
    for (int i = 0; i < queryDesctiptorSize; i++) {
        double distance = allMatches[i].distance;
        if (distance < minDistance) {
            minDistance = distance;
        }
        if (distance > maxDistance) {
            maxDistance = distance;
        }
    }
//    printf("Maximum distance: %f \n", maxDistance);
//    printf("Minimum distance: %f \n", minDistance);
    
    // Draw only good matches (i.e. whose distance is less than 3 * minDistance)
    for (int i = 0; i < queryDesctiptorSize; i++) {
        if (allMatches[i].distance < 3 * minDistance) {
            goodMatches.push_back(allMatches[i]);
        }
    }
}

void queryImageCorners(Mat& queryImage, vector<Point2f>& queryImageCorners)
{
    queryImageCorners[0] = cvPoint(0, 0);
    queryImageCorners[1] = cvPoint(queryImage.cols, 0);
    queryImageCorners[2] = cvPoint(queryImage.cols, queryImage.rows);
    queryImageCorners[3] = cvPoint(0, queryImage.rows);
}

void drawLinesBetweenCornersInSceneImage(Mat& sceneImage, vector<Point2f>& sceneCorners, int xOffset, int yOffset)
{
    line(sceneImage, Point2f(sceneCorners[0].x + xOffset, sceneCorners[0].y + yOffset), Point2f(sceneCorners[1].x + xOffset, sceneCorners[1].y + yOffset), Scalar(0, 255, 0), 4);
    line(sceneImage, Point2f(sceneCorners[1].x + xOffset, sceneCorners[1].y + yOffset), Point2f(sceneCorners[2].x + xOffset, sceneCorners[2].y + yOffset), Scalar(0, 255, 0), 4);
    line(sceneImage, Point2f(sceneCorners[2].x + xOffset, sceneCorners[2].y + yOffset), Point2f(sceneCorners[3].x + xOffset, sceneCorners[3].y + yOffset), Scalar(0, 255, 0), 4);
    line(sceneImage, Point2f(sceneCorners[3].x + xOffset, sceneCorners[3].y + yOffset), Point2f(sceneCorners[0].x + xOffset, sceneCorners[0].y + yOffset), Scalar(0, 255, 0), 4);
}

void drawLinesBetweenCornersInImage(Mat& image, vector<Point2f>& corners, int offsetInX, Scalar color)
{
    line(image, corners[0] + Point2f(offsetInX, 0), corners[1] + Point2f(offsetInX, 0), color, 4);
    line(image, corners[1] + Point2f(offsetInX, 0), corners[2] + Point2f(offsetInX, 0), color, 4);
    line(image, corners[2] + Point2f(offsetInX, 0), corners[3] + Point2f(offsetInX, 0), color, 4);
    line(image, corners[3] + Point2f(offsetInX, 0), corners[0] + Point2f(offsetInX, 0), color, 4);
}

void showImage(Mat& image)
{
    imshow("Image", image);
    waitKey(0);
}

void homographyForQueryInScene(vector<DMatch>& goodMatches, vector<KeyPoint>& queryKeypoints, vector<KeyPoint>& sceneKeypoints, Mat& homography)
{
    vector<Point2f> query, scene;
    for (int i = 0; i < goodMatches.size(); i++) {
        query.push_back(queryKeypoints[goodMatches[i].queryIdx].pt);
        scene.push_back(sceneKeypoints[goodMatches[i].trainIdx].pt);
    }
    homography = findHomography(query, scene, RANSAC);
}

void detectKeypointsInImage(Mat& image, vector<KeyPoint>& keypoints)
{
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    detector.detect(image, keypoints);
}

void calculateDescriptorsForImageAndKeypoints(Mat& image, vector<KeyPoint>& keypoints, Mat& descriptor)
{
    SurfDescriptorExtractor extractor;
    extractor.compute(image, keypoints, descriptor);
}

void findMatches(Mat& queryDescriptor, Mat& sceneDescriptor, vector<DMatch>& matches)
{
    FlannBasedMatcher matcher;
    if (sceneDescriptor.rows != 0 && sceneDescriptor.cols != 0) {
        matcher.match(queryDescriptor, sceneDescriptor, matches);
    }
}

bool queryObjectWasFound(vector<Point2f>& queryCorners, vector<Point2f>& objectCorners)
{
    bool wasQueryObjectFound = true;
    double acceptableError = 0.8;
    
    double queryUpSideLength = sqrt(pow(queryCorners[0].x - queryCorners[1].x, 2.0) + pow(queryCorners[0].y - queryCorners[1].y, 2.0));
    double queryRightSideLength = sqrt(pow(queryCorners[1].x - queryCorners[2].x, 2.0) + pow(queryCorners[1].y - queryCorners[2].y, 2.0));
    double queryDownSideLength = sqrt(pow(queryCorners[2].x - queryCorners[3].x, 2.0) + pow(queryCorners[2].y - queryCorners[3].y, 2.0));
    double queryLeftSideLength = sqrt(pow(queryCorners[3].x - queryCorners[0].x, 2.0) + pow(queryCorners[3].y - queryCorners[0].y, 2.0));
    
    double objectUpSideLength = sqrt(pow(objectCorners[0].x - objectCorners[1].x, 2.0) + pow(objectCorners[0].y - objectCorners[1].y, 2.0));
    double objectRightSideLength = sqrt(pow(objectCorners[1].x - objectCorners[2].x, 2.0) + pow(objectCorners[1].y - objectCorners[2].y, 2.0));
    double objectDownSideLength = sqrt(pow(objectCorners[2].x - objectCorners[3].x, 2.0) + pow(objectCorners[2].y - objectCorners[3].y, 2.0));
    double objectLeftSideLength = sqrt(pow(objectCorners[3].x - objectCorners[0].x, 2.0) + pow(objectCorners[3].y - objectCorners[0].y, 2.0));
    
    double upRatio = queryUpSideLength / objectUpSideLength;
    double rightRatio = queryRightSideLength / objectRightSideLength;
    double downRatio = queryDownSideLength / objectDownSideLength;
    double leftRatio = queryLeftSideLength / objectLeftSideLength;
    
//    printf("upRatio: %f\n", upRatio);
//    printf("rightRatio: %f\n", rightRatio);
//    printf("downRatio: %f\n", downRatio);
//    printf("leftRatio: %f\n", leftRatio);
//    printf("-------------------------\n");
    
    if (isnan(upRatio) || isnan(rightRatio) || isnan(leftRatio) || isnan(downRatio)) {
        return false;
    }
    if (isinf(upRatio) || isinf(rightRatio) || isinf(leftRatio) || isinf(downRatio)) {
        return false;
    }
    if (upRatio > rightRatio + acceptableError || upRatio < rightRatio - acceptableError) {
        return false;
    }
    if (upRatio > downRatio + acceptableError || upRatio < downRatio - acceptableError) {
        return false;
    }
    if (upRatio > leftRatio + acceptableError || upRatio < leftRatio - acceptableError) {
        return false;
    }
    
    return wasQueryObjectFound;
}

int main(int argc, const char *argv[])
{
    
    // Loading quary image and scene image
    Mat queryImage = imread("/Users/AleksanderGrzyb/Documents/Studia/Semestr_8/Przetwarzanie_i_Rozpoznawanie_Obrazow/Programy/PiROFinalProject/SampleImages/Newspapers/object.JPG");
    Mat sceneImage = imread("/Users/AleksanderGrzyb/Documents/Studia/Semestr_8/Przetwarzanie_i_Rozpoznawanie_Obrazow/Programy/PiROFinalProject/SampleImages/Newspapers/sample2.JPG");
    
    // Resizing
    resize(queryImage, queryImage, Size(queryImage.size().width * 0.3, queryImage.size().height * 0.3));
    resize(sceneImage, sceneImage, Size(sceneImage.size().width * 0.3, sceneImage.size().height * 0.3));
    
    // Constant values
    float windowRatio = 0.4;
    float xStepRatio = 0.1;
    float yStepRatio = 0.1;
    
    int xStep = sceneImage.cols * xStepRatio;
    int yStep = sceneImage.rows * yStepRatio;
    
    int windowWidth = queryImage.cols * windowRatio;
    int windowHeight = queryImage.rows * windowRatio;
    
    // Keypoints and descriptor of query image
    vector<KeyPoint> queryKeypoints;
    Mat queryDescriptor;
    detectKeypointsInImage(queryImage, queryKeypoints);
    calculateDescriptorsForImageAndKeypoints(queryImage, queryKeypoints, queryDescriptor);
    
    // Getting corners of query image
    vector<Point2f> queryCorners(4);
    queryImageCorners(queryImage, queryCorners);
    
    // Sliding window image, descriptor, keypoints
    vector<KeyPoint> windowKeypoints;
    Mat windowImage, windowDescriptor;
    
    // Data structures for matches
    vector<DMatch> allMatches, goodMatches;
    
    // Visualization of matches and found objects
    Mat matchImage, homography;
    vector<Point2f> objectCorners(4);
    
    for (int y = 0; y < sceneImage.rows - windowHeight - 1; y = y + yStep) {
        for (int x = 0; x < sceneImage.cols - windowWidth - 1; x = x + xStep) {
            windowImage = sceneImage(Rect(x, y, windowWidth, windowHeight));
            detectKeypointsInImage(windowImage, windowKeypoints);
            calculateDescriptorsForImageAndKeypoints(windowImage, windowKeypoints, windowDescriptor);
            findMatches(queryDescriptor, windowDescriptor, allMatches);
            if (allMatches.size() > 0) {
                sortMatchesToFindGoodOnes(allMatches, queryDescriptor.rows, goodMatches);
                drawMatches(queryImage, queryKeypoints, windowImage, windowKeypoints, goodMatches, matchImage);
                homographyForQueryInScene(goodMatches, queryKeypoints, windowKeypoints, homography);
                if (homography.cols != 0 && homography.rows != 0) {
                    perspectiveTransform(queryCorners, objectCorners, homography);
                    bool objectWasFound = queryObjectWasFound(queryCorners, objectCorners);
                    if (objectWasFound) {
                        drawLinesBetweenCornersInImage(matchImage, objectCorners, queryImage.cols, Scalar(0, 255, 0));
                        drawLinesBetweenCornersInSceneImage(sceneImage, objectCorners, x, y);
                    }
                    else {
                        drawLinesBetweenCornersInImage(matchImage, objectCorners, queryImage.cols, Scalar(255, 0, 0));
                    }
                }
            }
            windowKeypoints.clear(); allMatches.clear(); goodMatches.clear(); objectCorners.clear();
        }
        windowKeypoints.clear(); allMatches.clear(); goodMatches.clear(); objectCorners.clear();
    }
    showImage(sceneImage);
    return 0;
}

