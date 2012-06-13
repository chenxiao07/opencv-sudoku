#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
 
using namespace cv; 

void Reverse(Mat& src) {
    for(int y=0; y<src.size().height; y++) {
        uchar *row = src.ptr(y);
        for(int x=0; x<src.size().width; x++) {
            row[x] = 255 - row[x];
        }
    }
}
 
void PerspectiveCut(const Mat& src, Mat& dst, const Point2f src_point[]) {

    float width = dst.size().width;
    float height = dst.size().height;
    Point2f a(0.0f, 0.0f), b(width, 0.0f), c(0.0f, height), d(width, height);
    Point2f dst_point[] = {a, b, c, d};
    Mat M = getPerspectiveTransform(src_point, dst_point);

    warpPerspective(src, dst, M, dst.size());
}

void FindContours(const Mat& src, Point2f& a, Point2f& b, Point2f& c, Point2f& d) {
    int width = src.size().width;
    int height = src.size().height;
    Point2i ra(0, 0), rb(width, 0), rc(0, height), rd(width, height);
    double i=9999999999.0f;
    double da=i, db=i, dc=i, dd=i;
    for(int y=0; y<height; y++) {
        const uchar *row = src.ptr(y);
        for(int x=0; x<width; x++) {
            if(row[x]!=0)
            {
                Point2i temp(x, y);
                if(norm(ra - temp) < da) {
                    a.x = x;
                    a.y = y;
                    da = norm(ra - temp);
                }
                if(norm(rb - temp) < db) {
                    b.x = x;
                    b.y = y;
                    db = norm(rb - temp);
                }
                if(norm(rc - temp) < dc) {
                    c.x = x;
                    c.y = y;
                    dc = norm(rc - temp);
                }
                if(norm(rd - temp) < dd) {
                    d.x = x;
                    d.y = y;
                    dd = norm(rd - temp);
                }
            }
        }
    }
}

//this function goes over the image and finds the maximum x, and y of non-zero pixels
Rect shGetBoundingRect(Mat inputMat){
    int maxX = 0;
    int minX = inputMat.size().width;
    int maxY = 0;
    int minY = inputMat.size().height;
    int pixelCnt = 0;
        for(int y=0;y<inputMat.size().height;y++)
        {
            uchar *row = inputMat.ptr(y);
            for(int x=0;x<inputMat.size().width;x++)
            {
                if(row[x]!=0)
                {
                    pixelCnt++;
                    if (x > maxX)    maxX = x;
                    if (y > maxY)    maxY = y;
                    if (x < minX)    minX = x;
                    if (y < minY)    minY = y;
                }
            }
        }
        printf("the pixel count is: %d\n", pixelCnt);
        if (pixelCnt < 10){          //if there is just noise don't return a valid rect
            return Rect(0, 0, -10, -10);
        }
        else{
            return Rect(minX, minY, maxX-minX, maxY-minY);
        }
}
void shXscan(Mat inputMat, int *x2, int *x4, int *x6, int *lastYline, int a, int b, int c, int cutCountGoal){
        int currColor = 0;
        int cutCount = 0;               //number of lines cut in this scan
        int cutCount2 = 0;
        int currColor2 = 0;
        //int x2 = 0;
        //int x4 = 0;
        //int x6 = 0;           //points to do flood fill
        int sucessCount = 0;
        //int lastYline = 0;
        #define MIN_LINE_WIDTH = 2;     //number of pixels to define a line
        for(int y=0;y<inputMat.size().height;y++)
        {
            uchar *row = inputMat.ptr(y);
            currColor = row[0];
            for(int x=1;x<inputMat.size().width-1;x++)
            {
                if((row[x] != currColor) && (row[x+1] != currColor))
                {
                    cutCount++;
                    currColor = row[x];
                }
            }
            //printf("line %d, cuts %d lines\n", y, cutCount);
            if (cutCount == cutCountGoal){
                sucessCount++;
            }
            if (sucessCount > 4 ){//rescan that line and fill with grey
                cutCount2 = 0;
                currColor2 = row[0];
                for(int x=1;x<inputMat.size().width-1;x++)
                {
                    if((row[x] != currColor2) && (row[x+1] != currColor2))
                    {
                        cutCount2++;
                        currColor2 = row[x];
                    }
                    //printf("pos: %d, cutcount2: %d\n", x, cutCount2);
                    if (cutCount2 == a){
                        *x2 = x;    //this is the last pixel where cutCount was 1
                    }
                    if (cutCount2 == b){
                        *x4 = x;        //this is the last pixel where cutCount was 3
                    }
                    if (cutCount2 == c){
                        *x6 = x;        //this is the last pixel where cutCount was 1
                    }
                }
                //line(outerBox, Point(x4,y), Point(x6,y), CV_RGB(255,255,64), 5);
                *lastYline = y;
                break;
            }
            cutCount = 0;
        }
        if (sucessCount == 0){
            printf("could not find enough cut lines to parse!\n");
        }
}
 
//the code below detects the point that fills the largest area
//this point is returned, the inputMatrix is alterned in the process
Point shLargestFlood(Mat inputMat){
    printf("inside largestFlood");
    int count=0;
    int max=-1;
    Point maxPt;
 
    for(int y=0;y<inputMat.size().height;y++)
    {
        uchar *row = inputMat.ptr(y);
        for(int x=0;x<inputMat.size().width;x++)
        {
            if(row[x]>=128)
            {
                int area = floodFill(inputMat, Point(x,y), CV_RGB(0,0,64));
                printf("filling %d, %d gray\n", x, y);
                if(area>max)
                {
                    maxPt = Point(x,y);
                    max = area;
                }
            }
        }
    }
    return maxPt;
}
 
void shGetRidOfColor(Mat inputMat, int color){
        //this loop goes over the image and changes any shade equal to color
        //to 0.
        for(int y=0;y<inputMat.size().height;y++)
        {
            uchar *row = inputMat.ptr(y);
            for(int x=0;x<inputMat.size().width;x++)
            {
                if(row[x]==color)
                {
                    int area = floodFill(inputMat, Point(x,y), CV_RGB(0,0,0));
                }
            }
        }
}
 
void shParse3x3(Mat mask3x3, Mat origSmall, int zoneNum){
    char imgLabel[50];
    int notUsed = 0;
    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    Mat kernel2 = (Mat_<uchar>(3,3) << 1,1,1,1,1,1,1,1,1);
    int xLoc[3] = {0,0,0};
    int lastYline = 0;
    Rect finalBB;
    Mat tempBuffer;
    Mat finalMat;
 
    Mat easel = mask3x3.clone();        //make a copy of the mask
    bitwise_and(mask3x3, origSmall, mask3x3);
    dilate(mask3x3, mask3x3, kernel, Point(-1,-1),2);
    threshold(mask3x3,mask3x3,40,255,THRESH_BINARY);  //this may not be needed, as the image is already white
    Point maxPtZone3 = shLargestFlood(mask3x3);
    int area2 = floodFill(mask3x3, maxPtZone3, CV_RGB(255,255,255)); //color grid white
    shGetRidOfColor(mask3x3, 64);
    threshold(easel,easel,40,255,THRESH_BINARY);  //make zone3 all white
    erode(easel,easel,kernel2,Point(-1,-1),10);
    subtract(easel, mask3x3, easel);
    Mat zone3template = easel.clone();
    for (int j=0; j<3; j++){
        //need to scan for zones, they will be the first scan line that is cut three white regions
        //when the zones are aquired we then have all the regions in the puzzle isolated
        shXscan(zone3template, &xLoc[0], &xLoc[1], &xLoc[2], &lastYline, 0, 2, 4, 6);
 
        //now that we have the zones we need to get all the characters
        for (int k=0; k<3; k++){
            easel = zone3template.clone();
            //printf("the flood point is x:%d, y:%d\n", xLoc[k], lastYline);
            area2 = floodFill(easel, Point(xLoc[k]+1,lastYline), CV_RGB(255,255,64));       //fill in with gray
            shGetRidOfColor(easel, 255);    //get rid of white sections but leave gray
            area2 = floodFill(easel, Point(xLoc[k]+1,lastYline), CV_RGB(255,255,255));      //fill in with white to give the mask of one cell
            bitwise_and(easel, origSmall, easel);           //mask out just one number -- yes!
            finalBB = shGetBoundingRect(easel);//get bounding box
            //printf("the BB width is %d \n", finalBB.width);
            if (finalBB.width > 0){          //if it's less than 0 then there is nothing in the box
                tempBuffer = easel(finalBB);
                finalMat = tempBuffer.clone();
                notUsed = sprintf (imgLabel, "easel%d_%d_%d", zoneNum, k, j);
                imshow(imgLabel,finalMat);
            }
            else{                   //code to report this cell is a 0
            }
        }
        //fill all the areas white so the rescan won't pick them up
        area2 = floodFill(zone3template, Point(xLoc[0]+1,lastYline), CV_RGB(0,0,0));
        area2 = floodFill(zone3template, Point(xLoc[1]+1,lastYline), CV_RGB(0,0,0));
        area2 = floodFill(zone3template, Point(xLoc[2]+1,lastYline), CV_RGB(0,0,0));
    }
}
