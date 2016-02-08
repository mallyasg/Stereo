#include "Stereo.hpp"

Stereo::Stereo() {
}

Stereo::Stereo(
    cv::Mat leftCameraMatrix,
    cv::Mat rightCameraMatrix,
    cv::Mat leftDistCoeffs,
    cv::Mat rightDistCoeffs
    ) {
  leftCameraMatrix.copyTo(mLeftCameraMatrix);
  rightCameraMatrix.copyTo(mRightCameraMatrix);
  leftDistCoeffs.copyTo(mLeftDistCoeffs);
  rightDistCoeffs.copyTo(mRightDistCoeffs);
}

void Stereo::detectChessBoardCorners(
    const std::vector<cv::Mat> &leftImages,
    const std::vector<cv::Mat> &rightImages,
    std::vector<std::vector<cv::Point2f> > &leftImagePoints,
    std::vector<std::vector<cv::Point2f> > &rightImagePoints,
    std::vector<cv::Mat> goodLeftImages,
    std::vector<cv::Mat> goodRightImages,
    cv::Size boardSize,
    bool displayCorners
    ) {
  // Step 1 : Check if the left and right image sizes are the same
  assert(leftImages.size() == rightImages.size());
  
  // Step 2 : Iterate through all the images to compute the chess board corners
  for (size_t i = 0; i < leftImages.size(); i++) {
    
    std::vector<cv::Point2f> leftCorners;
    // Step 2a : Detect the chess board corners in the left image
    bool foundLeft = cv::findChessboardCorners(
        leftImages[i], 
        boardSize, 
        leftCorners, 
        cv::CALIB_CB_ADAPTIVE_THRESH | \
        cv::CALIB_CB_NORMALIZE_IMAGE
        );
    // Step 2b : If chess board corners were detected, refine the chess board corner position
    if (foundLeft) {
      cv::Size zeroZone = cv::Size(-1, -1);
      cv::Size winSize = cv::Size(11, 11);
      cv::TermCriteria termCriteria = cv::TermCriteria(
          cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
          30, 
          0.01
          );

      cv::cornerSubPix(
          leftImages[i], 
          leftCorners, 
          winSize, 
          zeroZone, 
          termCriteria
          );
    }
    
    std::vector<cv::Point2f> rightCorners;
    // Step 2c : Detect chess board corners in the right image
    bool foundRight = cv::findChessboardCorners(
        rightImages[i], 
        boardSize, 
        rightCorners, 
        cv::CALIB_CB_ADAPTIVE_THRESH | \
        cv::CALIB_CB_NORMALIZE_IMAGE
        );
    // Step 2d : If chess board corners were found in the right image refine the position of the 
    // corners.
    if (foundRight) {
      cv::Size zeroZone = cv::Size(-1, -1);
      cv::Size winSize = cv::Size(11, 11);
      cv::TermCriteria termCriteria = cv::TermCriteria(
          cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
          30, 
          0.01
          );

      cv::cornerSubPix(
          rightImages[i], 
          rightCorners, 
          winSize, 
          zeroZone, 
          termCriteria
          );
    }
    // Step 2e : If corneres were detected in both left and right images, then save 
    // the corner positions to the vector and push the images onto a vector of good
    // images
    if (foundLeft && foundRight) {
      leftImagePoints.push_back(leftCorners);
      rightImagePoints.push_back(rightCorners);
      goodLeftImages.push_back(leftImages[i]);
      goodRightImages.push_back(rightImages[i]);
    }

    // Step 2f : If display corners is True then show the images with corners detected
    if (displayCorners) {
      cv::Mat leftImage, rightImage;
      cv::cvtColor(leftImages[i], leftImage, cv::COLOR_GRAY2BGR);
      cv::cvtColor(rightImages[i], rightImage, cv::COLOR_GRAY2BGR);
      cv::drawChessboardCorners(leftImage, boardSize, leftCorners, foundLeft);
      cv::drawChessboardCorners(rightImage, boardSize, rightCorners, foundRight);
      cv::waitKey(33);
    }
  }
}

void Stereo::generateObjectPoints(
    cv::Size boardSize,
    std::vector<std::vector<cv::Point3f> > &objectPoints,
    int squareSize,
    int numGoodImages
    ) {
  // NOTE1 :- The square size should be in cms
  // NOTE2 :- The board size defines the number of interior chess 
  //          board corners

  // Step 1 : Store the 3D co-ordinates of the chess board corners
  for (int i = 0; i < numGoodImages; ++i) {
    std::vector<cv::Point3f> tempObjectPoints;
    for (int j = 0; j < boardSize.height; ++j) {
      for (int k = 0; k < boardSize.width; ++k) {
        tempObjectPoints.push_back(cv::Point3f(k * squareSize, j * squareSize, 0));
      }
    }
    objectPoints.push_back(tempObjectPoints);
  }
}

void Stereo::calibrate(
    const std::vector<cv::Mat> &leftImages,
    const std::vector<cv::Mat> &rightImages,
    cv::Size boardSize,
    int squareSize,
    cv::Size imageSize
    ) {

  std::vector<cv::Mat> goodLeftImages;
  std::vector<cv::Mat> goodRightImages;
  
  std::vector<std::vector<cv::Point2f> > leftImagePoints;
  std::vector<std::vector<cv::Point2f> > rightImagePoints;
  // Step 1 : Detect the chessboard corners in left and right images
  detectChessBoardCorners(
      leftImages,
      rightImages,
      leftImagePoints,
      rightImagePoints,
      goodLeftImages,
      goodRightImages,
      boardSize,
      true
      );

  // Step 2 : Using the good images (i.e. images in which corners can 
  //          be detected) generate a vector of object points
  int numGoodImages = goodLeftImages.size();
  std::vector<std::vector<cv::Point3f> > objectPoints;
  generateObjectPoints(
      boardSize,
      objectPoints,
      squareSize,
      numGoodImages
      );
  // Step 3 : Check if the camera matrix and distortion co-efficients are
  //          initialized. If they are not then use the opencv initCameraMatrix2D
  //          to initialize the left and right camera intrinsics.
  int flags = cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5;
  if (mLeftCameraMatrix.empty()) {
    mLeftCameraMatrix = cv::initCameraMatrix2D(objectPoints, leftImagePoints, imageSize, 0);
    flags += cv::CALIB_USE_INTRINSIC_GUESS;
  }

  if (mRightCameraMatrix.empty()) {
    mRightCameraMatrix = cv::initCameraMatrix2D(objectPoints, rightImagePoints, imageSize, 0);
    flags += cv::CALIB_USE_INTRINSIC_GUESS;
  }
  
  cv::TermCriteria termCriteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
      100,
      1e-5
      );

  // Step 4 : Perform Stereo Calibration
  double rms = cv::stereoCalibrate(
      objectPoints,
      leftImagePoints,
      rightImagePoints,
      mLeftCameraMatrix,
      mLeftDistCoeffs,
      mRightCameraMatrix,
      mRightDistCoeffs,
      imageSize,
      mRotationMatrix,
      mTranslationVector,
      mEssentialMatrix,
      mFundamentalMatrix,
      flags,
      termCriteria
      );

  std::cout << "Stereo Calibration performed with RMS Error : " << rms << std::endl;
  // Step 5 : Get the new Camera Matrix
  
  mNewLeftCameraMatrix = cv::getOptimalNewCameraMatrix(
      mLeftCameraMatrix,
      mLeftDistCoeffs,
      imageSize,
      1,
      imageSize
      );

  mNewRightCameraMatrix = cv::getOptimalNewCameraMatrix(
      mRightCameraMatrix,
      mRightDistCoeffs,
      imageSize,
      1,
      imageSize
      );

  // Step 6 : Check the quality of calibration
  double error = 0;
  int totalNumPoints = 0;
  std::vector<cv::Vec3f> leftLine;
  std::vector<cv::Vec3f> rightLine;
  cv::Mat leftMap1, leftMap2;
  cv::Mat rightMap1, rightMap2;

  for (int i = 0; i < numGoodImages; ++i) {
    int numPoints = leftImagePoints[i].size();
    cv::Mat leftPoints = cv::Mat(leftImagePoints[i]);
    cv::Mat rightPoints = cv::Mat(rightImagePoints[i]);
    
    cv::undistortPoints(
        leftImagePoints[i],
        leftImagePoints[i],
        mLeftCameraMatrix,
        mLeftDistCoeffs,
        cv::Mat(),
        mLeftCameraMatrix
      );

    cv::undistortPoints(
        rightImagePoints[i],
        rightImagePoints[i],
        mRightCameraMatrix,
        mRightDistCoeffs,
        cv::Mat(),
        mNewRightCameraMatrix
        );
    
    cv::computeCorrespondEpilines(
        leftImagePoints[i], 
        1, 
        mFundamentalMatrix, 
        leftLine
        );

    cv::computeCorrespondEpilines(
        rightImagePoints[i], 
        1, 
        mFundamentalMatrix, 
        rightLine
        );

    for (int j = 0; j < numPoints; ++j) {
      double errij = fabs(
          leftImagePoints[i][j].x * leftLine[j][0] + \
          leftImagePoints[i][j].y * leftLine[j][1] + \
          leftLine[j][2]
          ) + \
        fabs(
            rightImagePoints[i][j].x * rightLine[j][0] + \
            rightImagePoints[i][j].y * rightLine[j][1] + \
            rightLine[j][2]
            );
      error += errij;
    }
    totalNumPoints += numPoints;
  }
  std::cout << "Average epipolar error = " << error / totalNumPoints;
  std::cout << std::endl;
  // Step 7 : Rectify 
  cv::Mat Q;
  cv::Rect validROI1, validROI2;

  cv::stereoRectify(
      mLeftCameraMatrix, 
      mLeftDistCoeffs, 
      mRightCameraMatrix, 
      mRightDistCoeffs, 
      imageSize, 
      mRotationMatrix, 
      mTranslationVector, 
      mLeftRotation, 
      mRightRotation, 
      mLeftTranslation, 
      mRightTranslation,
      Q, 
      cv::CALIB_ZERO_DISPARITY, 
      1, 
      imageSize, 
      &validROI1, 
      &validROI2
      );
  cv::initUndistortRectifyMap(
      mLeftCameraMatrix, 
      mLeftDistCoeffs, 
      mLeftRotation, 
      mLeftTranslation, 
      imageSize, 
      CV_16SC2, 
      mRMapLeft[0],
      mRMapLeft[1]
      );

  cv::initUndistortRectifyMap(
      mRightCameraMatrix, 
      mRightDistCoeffs, 
      mRightRotation, 
      mRightTranslation, 
      imageSize, 
      CV_16SC2, 
      mRMapRight[0],
      mRMapRight[1]
      );

  for (int i = 0; i < numGoodImages; ++i) {
    cv::Mat rectifiedLeftImage, colorLeftImage;
    cv::Mat rectifiedRightImage, colorRightImage;

    cv::remap(
        goodLeftImages[i], 
        rectifiedLeftImage, 
        mRMapLeft[0], 
        mRMapLeft[1], 
        cv::INTER_LINEAR
        );

    cv::remap(
        goodRightImages[i], 
        rectifiedRightImage, 
        mRMapRight[0], 
        mRMapRight[1], 
        cv::INTER_LINEAR
        );

    cv::cvtColor(rectifiedLeftImage, colorLeftImage, cv::COLOR_GRAY2BGR);
    cv::cvtColor(rectifiedRightImage, colorRightImage, cv::COLOR_GRAY2BGR);

    cv::Mat canvas(
        cv::Size(
          colorLeftImage.cols + colorRightImage.cols, 
          colorLeftImage.rows
          ), 
        CV_8UC3
        );
    cv::Mat canvasLeftPart = canvas(
        cv::Rect(
          0, 
          0, 
          colorLeftImage.cols, 
          canvas.rows)
        );

    cv::Mat canvasRightPart = canvas(
        cv::Rect(
          colorLeftImage.cols, 
          0, 
          colorRightImage.cols, 
          canvas.rows
          )
        );

    colorLeftImage.copyTo(canvasLeftPart);
    colorRightImage.copyTo(canvasRightPart);
    cv::Rect vroiLeft(validROI1.x, validROI1.y, validROI1.width, validROI1.height);
    cv::rectangle(canvasLeftPart, vroiLeft, cv::Scalar(0, 0, 255), 3, 8);

    cv::Rect vroiRight(validROI2.x, validROI2.y, validROI2.width, validROI2.height);
    cv::rectangle(canvasRightPart, vroiRight, cv::Scalar(0, 0, 255), 3, 8);

    for (int j = 0; j < canvas.rows; j += 16) {
      cv::line(canvas, cv::Point(j, 0), cv::Point(j, canvas.rows), cv::Scalar(0, 255, 0), 1, 8);
    }
    
    cv::namedWindow("Rectified", cv::WINDOW_AUTOSIZE);
    cv::imshow("Rectified", canvas);
    cv::waitKey(0);
  }
}
