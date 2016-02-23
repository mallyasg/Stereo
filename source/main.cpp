#include "Stereo.hpp"
void readImageToVector(
    cv::FileStorage imageListFs, 
    std::vector<cv::Mat> &images
    ) {
  cv::FileNode n = imageListFs.getFirstTopLevelNode();
  if (n.type() != cv::FileNode::SEQ) {
    std::cout << "Image List file is not sequential.\n";
    exit(0);
  }
  cv::FileNodeIterator it       = n.begin();
  cv::FileNodeIterator it_end   = n.end();
  for (; it != it_end; it++) {
    cv::Mat tempImage = cv::imread(std::string(*it));
    cv::Mat tempGrayImage;
    cv::cvtColor(tempImage, tempGrayImage, cv::COLOR_BGR2GRAY);
    images.push_back(tempGrayImage);
  }
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "Insufficient number of arguments.\n";
    return -1;
  }  
  std::string leftImageList = std::string(argv[1]);
  std::string rightImageList = std::string(argv[2]);

  std::string leftCameraIntrinsicFilename = std::string(argv[3]);
  std::string rightCameraIntrinsicFilename = std::string(argv[4]);
  
  std::vector<cv::Mat> leftImages;
  std::vector<cv::Mat> rightImages;

  cv::Mat leftCameraMatrix;
  cv::Mat rightCameraMatrix;
  
  cv::Mat leftDistCoeffs;
  cv::Mat rightDistCoeffs;

  cv::Size boardSize = cv::Size(9, 6);
  int squareSize = 3;
  cv::Size imageSize = cv::Size(640, 480);

  // Step 1 : Load the left images
  cv::FileStorage leftImageFs(leftImageList, cv::FileStorage::READ);
  if (!leftImageFs.isOpened()) {
    std::cout << "Unable to open " << leftImageList << "." << std::endl;
    return -1;
  }
  readImageToVector(leftImageFs, leftImages);
  
  
  // Step 2 : Load the right images
  cv::FileStorage rightImageFs(rightImageList, cv::FileStorage::READ);
  if (!rightImageFs.isOpened()) {
    std::cout << "Unable to open " << rightImageList << "." << std::endl;
    return -1;
  }
  readImageToVector(rightImageFs, rightImages);
  
  if (leftImages.size() != rightImages.size()) {
    std::cout << "The number of images capatured in the left and ";
    std::cout << "right cameras must be the same." << std::endl;
    return -1;
  }
   
  //for (int i = 0; i < leftImages.size(); ++i) {
  //  cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
  //  cv::Mat compositeImage(
  //      cv::Size(
  //        leftImages[i].cols + rightImages[i].cols, 
  //        leftImages[i].rows
  //        ), 
  //      CV_8UC1
  //      );
  //  cv::Mat leftHalf(compositeImage, cv::Rect(0, 0, leftImages[i].cols, leftImages[i].rows));
  //  cv::Mat rightHalf(compositeImage, cv::Rect(leftImages[i].cols, 0, rightImages[i].cols, leftImages[i].rows));

  //  leftImages[i].copyTo(leftHalf);
  //  rightImages[i].copyTo(rightHalf);
  //  
  //  cv::imshow("Input Image", compositeImage);
  //  cv::waitKey(0);
  //}

  // Step 3 : Read the left camera intrinsics file.
  cv::FileStorage leftCameraFs(
      leftCameraIntrinsicFilename, 
      cv::FileStorage::READ
      );
  
  if (!leftImageFs.isOpened()) {
    std::cout << "Unable to open " << leftCameraIntrinsicFilename << ".\n";
    return -1;
  }
  
  // Step 4 : Load the camera matrix corresponding to left camera
  leftCameraFs["camera_matrix"] >> leftCameraMatrix;

  // Step 5 : Load the camera distortion co-efficients corresponding 
  //          to left camera
  leftCameraFs["distortion_coefficients"] >> leftDistCoeffs;
  
  // Step 6 : Read the right camera intrinsics file.
  cv::FileStorage rightCameraFs(
      rightCameraIntrinsicFilename,
      cv::FileStorage::READ
      );

  // Step 7 : Load the camera matrix corresponding to right camera
  rightCameraFs["camera_matrix"] >> rightCameraMatrix;

  // Step 8 : Load the camera distortion co-efficients coreesponding 
  //          to right camera
  rightCameraFs["distortion_coefficients"] >> rightDistCoeffs;

  Stereo stereoEngine(
      leftCameraMatrix,
      rightCameraMatrix,
      leftDistCoeffs,
      rightDistCoeffs
      );

  // Step 9 : Stereo calibrate
  stereoEngine.calibrate(
      leftImages,
      rightImages,
      boardSize,
      squareSize,
      imageSize
      );
  return 0;
}
