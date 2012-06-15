#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
 
using namespace cv; 
 
void shXscan(Mat inputMat, int *x2, int *x4, int *x6, int *lastYline, int a, int b, int c, int cutCountGoal);
Point shLargestFlood(Mat inputMat);
void shGetRidOfColor(Mat inputMat, int color);
Rect shGetBoundingRect(Mat inputMat);
void shParse3x3(Mat mask, Mat orig, int zoneNum);
void shMakeBorder(Mat inputMat, int thickN, int color);
void PerspectiveCut(const Mat& src, Mat& dst, const Point src_point[]);
void FindContours(const Mat& src, Point& a, Point& b, Point& c, Point& d);
void Reverse(Mat& src);
int CalSum(Mat& src);
void Clear(Mat& src);
void Shape_height(Mat& ref, Mat& src, Mat& dst);
void Shape_width(Mat& ref, Mat& src, Mat& dst);
void findRect(Mat& src, Mat& dst, Mat& show);
