#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include "opencv2/highgui/highgui.hpp"
#include "sudokuHelper.h"
#include "basicOCR.h"
 
using namespace cv;         //now using the OpenCV C++ interface, much more Matlabesque
 
int main(int argc, const char** argv)
{
        //Mat is a data type in the Open CV C++ interface
                //this loads the image from a file, and creates a Mat
        Mat sudoku = imread("IMG_0133half.jpg", 0);
        Mat splitImgArr[9][9];              //9x9 Matrix of images split off of original
 
        Mat outerBox = Mat(sudoku.size(), CV_8UC1);
        GaussianBlur(sudoku, outerBox, Size(11,11), 0);
 
        adaptiveThreshold(outerBox, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
        bitwise_not(outerBox, outerBox);
        Mat startImg = outerBox.clone();
        Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
        //dilate(outerBox, outerBox, kernel);

        //below is code for detecting the outer box, all items are colored 64 in the process
        Point maxPt = shLargestFlood(outerBox);

        //color the largest area white
        floodFill(outerBox, maxPt, CV_RGB(255,255,255));

        printf("Height=%d, Width=%d\n", outerBox.size().height, outerBox.size().width);
 
        //this loop goes over the image and colors all the grey areas black
        shGetRidOfColor(outerBox, 64);
 
        //dilate(outerBox, outerBox, kernel);
        Mat bigGrid = outerBox.clone();     //save a copy of just the big grid for later
 
        Point2f a(5.0f, 40.0f), b(220.0f, 40.0f), c(5.0f, 250.0f), d(220.0f, 250.0f);
        FindContours(outerBox, a, b, c, d);
        Point2f src_point[] = {a, b, c, d};
        PerspectiveCut(sudoku, bigGrid, src_point);

        circle(outerBox, a, 15, CV_RGB(66,66,66));
        circle(outerBox, b, 15, CV_RGB(66,66,66));
        circle(outerBox, c, 15, CV_RGB(66,66,66));
        circle(outerBox, d, 15, CV_RGB(66,66,66));

        adaptiveThreshold(bigGrid, bigGrid, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 9);
        Reverse(bigGrid);
        dilate(bigGrid, bigGrid, kernel);
        Reverse(bigGrid);

        imshow("outerBoxWorkingDemo:",outerBox);      //demo it's working
        imshow("PerspectiveCut:",bigGrid);      //demo it's working
        waitKey();

        return 0;
}
