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

void findGoodMatches(vector<DMatch>& allMatches, int queryDesctiptorSize, vector<DMatch>& goodMatches)
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
    printf("Maximum distance: %f \n", maxDistance);
    printf("Minimum distance: %f \n", minDistance);
    
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

void drawLinesBetweenCornersInImage(Mat& image, vector<Point2f>& corners, int offsetInX)
{
    line(image, corners[0] + Point2f(offsetInX, 0), corners[1] + Point2f(offsetInX, 0), Scalar(0, 255, 0), 4);
    line(image, corners[1] + Point2f(offsetInX, 0), corners[2] + Point2f(offsetInX, 0), Scalar(0, 255, 0), 4);
    line(image, corners[2] + Point2f(offsetInX, 0), corners[3] + Point2f(offsetInX, 0), Scalar(0, 255, 0), 4);
    line(image, corners[3] + Point2f(offsetInX, 0), corners[0] + Point2f(offsetInX, 0), Scalar(0, 255, 0), 4);
}

void showImage(Mat& image)
{
    imshow("Image", image);
    waitKey(0);
}

void homographyForQueryInScene(vector<DMatch>& goodMatches, vector<KeyPoint>& queryKeypoints, vector<KeyPoint>& sceneKeypoints, Mat& homography)
{
    vector<Point2f> query;
    vector<Point2f> scene;
    for (int i = 0; i < goodMatches.size(); i++) {
        query.push_back(queryKeypoints[goodMatches[i].queryIdx].pt);
        scene.push_back(sceneKeypoints[goodMatches[i].trainIdx].pt);
    }
    homography = findHomography(query, scene, RANSAC);
}

void detectKeypointsInImages(Mat& queryImage, Mat& sceneImage, vector<KeyPoint>& queryKeypoints, vector<KeyPoint>& sceneKeypoints)
{
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    detector.detect(queryImage, queryKeypoints);
    detector.detect(sceneImage, sceneKeypoints);
}

void calculateDescriptorsForKeypoints(Mat& queryImage, Mat& sceneImage, vector<KeyPoint>& queryKeypoints, vector<KeyPoint>& sceneKeypoints, Mat& queryDescriptor, Mat& sceneDescriptor)
{
    SurfDescriptorExtractor extractor;
    extractor.compute(queryImage, queryKeypoints, queryDescriptor);
    extractor.compute(sceneImage, sceneKeypoints, sceneDescriptor);
}

void findMatches(Mat& queryDescriptor, Mat& sceneDescriptor, vector<DMatch>& matches)
{
    FlannBasedMatcher matcher;
    matcher.match(queryDescriptor, sceneDescriptor, matches);
}

int main(int argc, const char *argv[])
{
    // Loading quary image and scene image
    Mat queryImage = imread("/Users/AleksanderGrzyb/Documents/Studia/Semestr_8/Przetwarzanie_i_Rozpoznawanie_Obrazow/Programy/PiROFinalProject/SampleImages/Newspapers/object.JPG");
    Mat sceneImage = imread("/Users/AleksanderGrzyb/Documents/Studia/Semestr_8/Przetwarzanie_i_Rozpoznawanie_Obrazow/Programy/PiROFinalProject/SampleImages/Newspapers/sample1.JPG");
    
    // Resizing images
    resize(queryImage, queryImage, Size(queryImage.size().width * 0.3, queryImage.size().height * 0.3));
    resize(sceneImage, sceneImage, Size(sceneImage.size().width * 0.3, sceneImage.size().height * 0.3));
    
    // Detect the keypoints using SURF Detector
    vector<KeyPoint> queryKeypoints, sceneKeypoints;
    detectKeypointsInImages(queryImage, sceneImage, queryKeypoints, sceneKeypoints);

    // Calculate descriptors
    Mat queryDescriptor, sceneDescriptor;
    calculateDescriptorsForKeypoints(queryImage, sceneImage, queryKeypoints, sceneKeypoints, queryDescriptor, sceneDescriptor);
    
    // Matching descriptor vectors using FLANN matcher
    vector<DMatch> matches;
    findMatches(queryDescriptor, sceneDescriptor, matches);
    
    // Finding good matches
    vector<DMatch> goodMatches;
    findGoodMatches(matches, queryDescriptor.rows, goodMatches);
    
    // Drawing matches
    Mat matchImage;
    drawMatches(queryImage, queryKeypoints, sceneImage, sceneKeypoints, goodMatches, matchImage);
    
    // Localazing query
    Mat homography;
    homographyForQueryInScene(goodMatches, queryKeypoints, sceneKeypoints, homography);
    
    // Getting corners of query image
    vector<Point2f> queryCorners(4);
    queryImageCorners(queryImage, queryCorners);
    
    // Transfor corners by given homography
    vector<Point2f> sceneCorners(4);
    perspectiveTransform(queryCorners, sceneCorners, homography);
    
    // Drawing lines between corners
    drawLinesBetweenCornersInImage(matchImage, sceneCorners, queryImage.cols);
    
    // Presenting image
    showImage(matchImage);
    return 0;
}

