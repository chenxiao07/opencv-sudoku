#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
 
using namespace cv; 
 
void shXscan(Mat inputMat, int *x2, int *x4, int *x6, int *lastYline, int a, int b, int c, int cutCountGoal);
Point shLargestFlood(Mat inputMat);
void shGetRidOfColor(Mat inputMat, int color);
Rect shGetBoundingRect(Mat inputMat);
void shParse3x3(Mat mask, Mat orig, int zoneNum);
void shMakeBorder(Mat inputMat, int thickN, int color);
void PerspectiveCut(const Mat& src, Mat& dst, const Point2f src_point[]);
void FindContours(const Mat& src, Point2f& a, Point2f& b, Point2f& c, Point2f& d);
void Reverse(Mat& src);
