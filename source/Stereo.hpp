#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

class Stereo {
  private:
    cv::Mat mLeftCameraMatrix;
    cv::Mat mRightCameraMatrix;
    cv::Mat mNewLeftCameraMatrix;
    cv::Mat mNewRightCameraMatrix;
    cv::Mat mLeftDistCoeffs;
    cv::Mat mRightDistCoeffs;
    cv::Mat mRotationMatrix;
    cv::Mat mTranslationVector;
    cv::Mat mEssentialMatrix;
    cv::Mat mFundamentalMatrix;
    cv::Mat mLeftRotation;
    cv::Mat mLeftTranslation;
    cv::Mat mRightRotation;
    cv::Mat mRightTranslation;
    cv::Mat mRMapLeft[2];
    cv::Mat mRMapRight[2];
  public:
    Stereo();
    Stereo(
        cv::Mat leftCameraMatrix,
        cv::Mat rightCameraMatrix,
        cv::Mat leftDistCoeffs,
        cv::Mat rightDistCoeffs
        );
    void calibrate(
        std::vector<cv::Mat> leftImages,
        std::vector<cv::Mat> rightImages
        );
  private:
    void detectChessBoardCorners( 
        const std::vector<cv::Mat> &leftImages,
        const std::vector<cv::Mat> &rightImages,
        std::vector<std::vector<cv::Point2f> > &leftImagePoints,
        std::vector<std::vector<cv::Point2f> > &rightImagePoints,
        std::vector<cv::Mat> goodLeftImages,
        std::vector<cv::Mat> goodRightImages,
        cv::Size boardSize,
        bool displayCorners
        );

    void generateObjectPoints(
        cv::Size boardSize,
        std::vector<std::vector<cv::Point3f> > &objectPoints,
        int squareSize,
        int numGoodImages
        );
};
