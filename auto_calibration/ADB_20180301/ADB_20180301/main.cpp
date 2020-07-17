//#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS
#include "omp.h" 
#include <iostream>
#include "time.h"
#include "math.h"
#include "opencv2/opencv.hpp"

/*Harris parameter setting*/
#define thresh_corner 106 //173
#define blockSize 2
#define apertureSize 3
#define k 0.04

#define color_range 16
#define max_mark_num 20000000

#define dist 10

/*Camera Parameter Input*/
#define Htar 1410 /* Target height*/
#define Dtar 4570 /* Target distance*/
#define Wtar 1120 /* Target width */
#define Hcam 2260 /* Camera height */
#define Ocam 0 /* Camera offset */
#define Pcam -1910 /* Camera position -:left +:right */
#define Xcam0 640 /* Xcam0 Center position(x) of camera*/
#define Ycam0 360 /* Ycam0 Center position(y) of camera */

cv::Rect rect;
using namespace cv;
bool drawing_rect = false;
bool selected = false;
void mouse_cb(int event, int x, int y, int flag, void* param);

/*Harris Corner Detector*/
Mat cornerHarris_demo(Mat src, Mat src_gray, Mat mark, int &mark_num, int &thr_corner_demo);
const char* source_window = "Source image";
const char* corners_window = "Corners detected";

/*Check Mark Detector*/
Mat check_mark(Mat src, Mat corner_img, Mat mark_coor, int &mark_num, int &ROI_x_start, int &ROI_x_end, int &ROI_y_start, int &ROI_y_end);
/*Decision Mark*/
int decision_mark(Mat src, int x, int y);

/*Get min and max*/
int min(int x, int y);
int max(int x, int y);

int main()
{
	Mat frame_one_gray;
	/*harris corner param setting*/
	Mat dst_norm_out;
	

	Mat harris_mark_coor = Mat::zeros(2 * max_mark_num, 1, CV_32FC1);
	Mat harris_img;
	int harris_mark_num,thr_corner = 106;
	int tar_dist = 4570, cam_h = 2260;
	
	/*====================ROI Input========================*/
	int x_start = 192  , x_end = 1094 ;
	int y_start = 111 , y_end = 515;
	/*printf("Please Input ROI x_start:");
	scanf_s("%d\n", &x_start);
	printf("Please Input ROI x_end:");
	scanf_s("%d\n", &x_end);
	printf("Please Input ROI y_start:");
	scanf_s("%d\n", &y_start);
	printf("Please Input ROI y_end:");
	scanf_s("%d\n", &y_end);
	printf("Finally ROI Start:[%d,%d] End:[%d,%d]\n", x_start, y_start, x_end, y_end);*/
	/////////////////////////////////////////////////////////

	/*Corner Decision*/
	Mat corner_result = Mat::zeros(2 * max_mark_num, 1, CV_32FC1);


	Mat frame_one;
	#ifdef    USE_CAMERA
		VideoCapture frame_capture_camera1(0);
		frame_capture_camera1.read(frame_one);
	#else 
		frame_one = imread("cali_0313.jpg");
	#endif
	
	FILE *fp;
	fp = fopen("calibration.txt", "r");
	if (!fp)
	{
		printf("no file\n");
	}
	
	{
		#ifdef USE_CAMERA
		if (!frame_capture_camera1.read(frame_one))
			break;
		#endif
		Size img_size = frame_one.size();
		cvtColor(frame_one, frame_one_gray, COLOR_BGR2GRAY);
		namedWindow(source_window);
		imshow(source_window, frame_one_gray);

		fscanf(fp, "%d %d %d %d %d %d %d", &thr_corner, &tar_dist, &cam_h, &x_start, &x_end, &y_start, &y_end);
	
		/*Check Corner color is correct or not*/
		Mat corner_img = Mat::zeros(1280, 720, CV_32FC1);
		harris_mark_num = 0;
		harris_img = cornerHarris_demo(frame_one, frame_one_gray, harris_mark_coor, harris_mark_num, thr_corner);
		printf("harris_mark_coor = (%f,%f) , harris_mark_num = %d\n", harris_mark_coor.at<float>(4, 0), harris_mark_coor.at<float>(5, 0), harris_mark_num);
		printf("%d\n", harris_img.dims);

		int index,input_x,input_y;
		int decision,corner_num = 0;
		for (int i = 0; i < harris_mark_num; i++)
		{
			index = i * 2;
			input_y =(int)harris_mark_coor.at<float>(index, 0);
			input_x =(int)harris_mark_coor.at<float>(index + 1, 0);
			decision = decision_mark(frame_one_gray, input_x, input_y);
			if (decision == 1)
			{
				corner_img.at<float>(input_x, input_y) = 1;
				printf("cor be = %f\n", corner_img.at<float>(input_x, input_y));
				corner_result.at<int>(2*corner_num, 0) = input_x;
				corner_result.at<int>(2*corner_num + 1, 0) = input_y;
				//printf("cor i = %d ,j = %d\n", corner_result.at<int>(2*corner_num, 0), corner_result.at<int>(2*corner_num+1, 0));
				corner_num++;

				//circle(frame_one_gray, Point(input_x, input_y), 5, Scalar(0), 2, 8, 0);
			}

		}
		/*printf("Result number = %d\n", corner_num);
		namedWindow(corners_window);*/
		imshow(corners_window, frame_one_gray);
		
		/*Decision and Filter Mark*/
		int i;
		int result_corner_num = 0;
		Mat result_corner_coor = Mat::zeros(2 * max_mark_num, 1, CV_32SC1);
		for (int xx = x_start; xx < x_end; xx++)
		{
			for (int yy = y_start; yy < y_end; yy++)
			{
				//printf("cor = %f\n", corner_img.at<float>(xx, yy));
				
				if (corner_img.at<float>(xx, yy) == 1)
				{
					for (i = 0; i < corner_num; i++)
					{
						int mark_x = result_corner_coor.at<int>(i * 2, 0);
						int mark_y = result_corner_coor.at<int>(i * 2+1, 0);
						//printf("mark i = %d\n", i);
						if (mark_x == 0 && mark_y == 0)
						{
							result_corner_coor.at<int>(i * 2, 0) = xx;
							result_corner_coor.at<int>(i * 2+1, 0) = yy;
							result_corner_num++;
							printf("num = %d , mark_x = %d , mark_y =%d\n", result_corner_num, result_corner_coor.at<int>(i * 2, 0), result_corner_coor.at<int>(i * 2+1, 0));
							break;
						}
						else if (abs(xx - mark_x) < dist && abs(yy - mark_y) < dist)
						{
							float result_harris = harris_img.at<float>(yy, xx);
							float temp_harris = harris_img.at<float>(mark_y, mark_x);
							if (result_harris > temp_harris)
							{
								result_corner_coor.at<int>(i * 2, 0) = xx;
								result_corner_coor.at<int>(i * 2 + 1, 0) = yy;
							}
							break;
						}
						else
						{
	
						}
					}
				}
			}
		}

		/*Camera Calibration Parameter*/

		int Dz = tar_dist + Ocam;

		int Xmarkl = result_corner_coor.at<int>(0, 0);
		int Xmarkr = result_corner_coor.at<int>(2*3, 0);
		int Ymarkl = result_corner_coor.at<int>(1, 0);
		int Ymarkr = result_corner_coor.at<int>(2*3+1, 0);

		int Xmark = Xmarkr - Xmarkl;
		int Ymark = Ymarkr - Ymarkl;

		int Xdiff = Pcam * Xmark / Wtar;
		int Ydiff = (Htar - cam_h) * Xmark / Wtar;
		Xmarkl += Xdiff;
		Ymarkl += Ydiff;
		int Xmark0 = Xmarkl + Xmark / 2;
		int Ymark0 = Ymarkl + Ymark / 2;

		int focus_pp, focus_ep;
		focus_pp = (int)(Xmark * Dz / Wtar);
		focus_ep = (double)(Xmark / 2) / atan((double)(Wtar)/Dz);
		double pitch = atan((double)(Ycam0 - Ymark0) / focus_pp);
		double pan = atan((double)(Xcam0 - Xmark0) / focus_ep);
		double roll = atan((double)-Ymark / Xmark);
		int X_foe = Xmark0;
		int Y_foe = Ymark0;
		printf("foe = (%d,%d)\n", X_foe, Y_foe);
		printf("focus = (%d,%d)\n", focus_pp, focus_ep);
		printf("pitch = %f\n", pitch);
		printf("pan = %f\n", pan);
		printf("roll = %f\n", roll);
		for (int i = 0; i < result_corner_num; i++)
		{
			int circle_pt_x = result_corner_coor.at<int>(2 * i, 0);
			int circle_pt_y = result_corner_coor.at<int>(2 * i + 1, 0);
			printf("circle pt_x = %d , pt_y = %d\n", circle_pt_x, circle_pt_y);
			circle(frame_one_gray, Point(circle_pt_x, circle_pt_y), 5, Scalar(0), 2, 8, 0);
		}



		namedWindow(corners_window);
		imshow(corners_window, frame_one_gray);
		waitKey(25);

	}
		system("pause");
	
}

int min(int x, int y)
{
	int result = -2;
	if (x > y)
	{
		result = y;
	}
	else{
		result = x;
	}
	return result;
}
int max(int x, int y)
{
	int result = -1;
	if (x > y)
	{
		result = x;
	}
	else{
		result = y;
	}
	return result;
}

Mat cornerHarris_demo(Mat src, Mat src_gray, Mat mark, int &mark_num,int &thr_corner_demo)
{
	Mat dst = Mat::zeros(src.size(), CV_32FC1);
	
	cornerHarris(src_gray, dst, blockSize, apertureSize, k);
	
	Mat dst_norm;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(dst_norm, corner_img);
	
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			//printf("dst %d\n", (int)dst_norm.at<float>(i, j));
			if ((int)dst_norm.at<float>(i, j) > thr_corner_demo)
			{
				//circle(src_gray, Point(j, i), 5, Scalar(0), 2, 8, 0);
				//printf("i = %d , j = %d\n",i,j);
				mark.at<float>(2*mark_num,0) = i;
				mark.at<float>(2 * mark_num + 1, 0) = j;
				mark_num++;
			}
		}
	}
	printf("mark_num = %d\n", mark_num);
	/*namedWindow(corners_window);
	imshow(corners_window, src_gray);*/
	return dst_norm;
}

int decision_mark(Mat src, int x, int y)
{
	int xx,yy;
	int range = color_range;
	int dis = 5;
	int mark_min[2], mark_max[2];
	for (int i = 0; i < 2; i++)
	{
		//printf("i = %d,x =%d,y=%d,row=%d,cols=%d\n", i, x, y, src.rows, src.cols);
		int px[4];
		xx = x - (dis + 1) * (i ? 1 : -1);
		yy = y - (dis + 1);
		if (yy >= (int)src.rows)
			yy = (int)src.rows - 1;
		if (xx >= (int)src.cols)
			xx = (int)src.cols - 1;
		if (xx < 0)
			xx = 0;
		if (yy < 0)
			yy = 0;
		//printf("xx = %d , yy = %d\n",xx,yy);
		px[0] = (int)src.at<uchar>(yy , xx);
		//px[0] = (int)src.at<uchar>(xx, yy);
		xx = x - (dis + 0) * (i ? 1 : -1);
		yy = y - (dis + 0);
		if (yy >= (int)src.rows)
			yy = (int)src.rows - 1;
		if (xx >= (int)src.cols)
			xx = (int)src.cols - 1;

		if (xx < 0)
			xx = 0;
		if (yy < 0)
			yy = 0;
		px[1] = src.at<uchar>(yy, xx);
		//px[1] = (int)src.at<uchar>(xx, yy);
		xx = x + (dis + 0) * (i ? 1 : -1);
		yy = y + (dis + 0);
		if (yy >= (int)src.rows)
			yy = (int)src.rows - 1;
		if (xx >= (int)src.cols)
			xx = (int)src.cols - 1;

		if (xx < 0)
			xx = 0;
		if (yy < 0)
			yy = 0;
		px[2] = src.at<uchar>(yy, xx);
		//px[2] = (int)src.at<uchar>(xx, yy);
		xx = x + (dis + 1) * (i ? 1 : -1);
		yy = y + (dis + 1);
		if (yy >= (int)src.rows)
			yy = (int)src.rows - 1;
		if (xx >= (int)src.cols)
			xx = (int)src.cols - 1;

		if (xx < 0)
			xx = 0;
		if (yy < 0)
			yy = 0;
		px[3] = src.at<uchar>(yy, xx);
		//px[3] = (int)src.at<uchar>(xx, yy);
		
		mark_min[i] = min(min(px[0], px[1]), min(px[2], px[3]));
		mark_max[i] = max(max(px[0], px[1]), max(px[2], px[3]));
		//printf("0 = %d ,1 = %d ,2 = %d ,3 =%d,max =%d,min= %d\n", px[0], px[1], px[2], px[3],mark_max[i], mark_min[i]);
	}
	if (mark_max[0] - mark_min[0]<range && mark_max[1] - mark_min[1]<range && (mark_min[0]>mark_max[1] || mark_min[1]>mark_max[0]))
	{
		return 1;
	}
	else{
		return 0;
	}
	

}

void mouse_cb(int event, int x, int y, int flag, void* param)
{
	cv::Mat *image = (cv::Mat*) param;
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_rect){
			rect.width = x - rect.x;
			rect.height = y - rect.y;
			printf("cas1 x=%d,y=%d,h=%d,w=%d\n", x, y, rect.height, rect.width);

			printf("case 1\n");
		}
		break;

	case CV_EVENT_LBUTTONDOWN:
		drawing_rect = true;
		rect = cv::Rect(x, y, 0, 0);
		printf("case2 x=%d,y=%d,h=%d,w=%d\n", rect.x, rect.y, rect.height, rect.width);
		printf("case 2\n");
		break;

	case CV_EVENT_LBUTTONUP:
		drawing_rect = false;
		if (rect.width < 0){
			rect.x += rect.width;
			rect.width *= -1;
		}
		if (rect.height < 0){
			rect.y += rect.height;
			rect.height *= -1;
		}
		cv::rectangle(*image, rect, cv::Scalar(0), 2);
		selected = true;
		printf("case 2\n");
		break;
	}

}