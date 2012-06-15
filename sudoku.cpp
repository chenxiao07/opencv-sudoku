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
        Mat sudoku = imread("sudoku-original.jpg", 0);
        int width = sudoku.size().width;
        int height = sudoku.size().height;
 
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
        Size2i s(180,180);
        Mat bigGrid = Mat(s, CV_8UC1);
        Mat outerGrid = Mat(s, CV_8UC1);
        Mat result = Mat(s, CV_8UC3);
        Mat mask = Mat(Size2i(width+2,height+2), CV_8UC1);
        findRect(outerBox, mask, sudoku);

        waitKey();

        return 0;
 
        Point2f a(5.0f, 40.0f), b(220.0f, 40.0f), c(5.0f, 250.0f), d(220.0f, 250.0f);
        //FindContours(outerBox, a, b, c, d);
        //Point2f src_point[] = {a, b, c, d};
        //PerspectiveCut(sudoku, bigGrid, src_point);
        //PerspectiveCut(outerBox, outerGrid, src_point);
        //Mat bigGrid2 = bigGrid.clone();
        //Shape_height(outerGrid, bigGrid2, bigGrid);
        //Shape_width(outerGrid, bigGrid2, bigGrid);
        imshow("bigGrid:",bigGrid);      //demo it's working
        imshow("outerGrid:",outerGrid);      //demo it's working

        adaptiveThreshold(bigGrid, bigGrid, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 9);
        //Reverse(bigGrid);
        //dilate(bigGrid, bigGrid, kernel);
        //Reverse(bigGrid);

        basicOCR ocr;

        cvtColor(bigGrid, result, CV_GRAY2BGR);

        for(int i=0; i<9; i++) {
            for(int j=0; j<9; j++) {
                Rect rect(i*20, j*20, 20, 20);
                Mat subimage = bigGrid(rect);
                if(CalSum(subimage)>22) {
                IplImage ipl_img = subimage;
                IplImage* img = &ipl_img;

                imshow("test", subimage);
                char* number = new char[10];
                sprintf(number, "%d", (int)ocr.classify(img, 0));
                putText(result,number,cvPoint(10+20*i, 10+20*j), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0,0,255));
                //rectangle(result, rect, cvScalar(0,0,255));
                imshow("sudoku:",result);      //demo it's working
                waitKey(100);
                }
            }
        }

        waitKey();

        return 0;
}
