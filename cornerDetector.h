#pragma once


#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "config.h"

using namespace cv;
using namespace std;

class cornerDetector
{
public:

	struct record
	{
		double t_detect;
		double t_gen_pro;
		double t_gen_res;
		double t_gen_sco;
		double t_nms;
		double minScore;
		double maxScore;
		Mat prototype[2][4];
		Mat response[2][4];
		Mat score;
		vector<Vec3d> scoreList;

		void reset()
		{
			t_detect = 0;
			t_gen_pro = 0;
			t_gen_res = 0;
			t_gen_sco = 0;
			t_nms = 0;
			minScore = INT_MAX;
			maxScore = INT_MIN;
			for (int m = 0; m < 2; m++)
			{
				for (int n = 0; n < 4; n++)
				{
					Mat m0,m1;
					prototype[m][n] = m0;
					response[m][n] = m1;					
				}
			}
			Mat m3;
			score = m3;
			scoreList.clear();
		}
	};

	//***计算角点响应***
	//compute the score of corner
	//-img: input, img
	//-r_pro: input, the radius of prototype
	//-r_nms: input, the radius of NMS
	//-t: input, the threshold of NMS
	void detecorCorner(Mat img,vector<Vec3d>& scoreList, int r_pro,int r_nms, int t = 0, record* addr_record = NULL);

private:

	int r = 0;
	Mat prototype[2][4];

	bool is_record = false;
	record* p_record;		// 指向detect的addr_record参数，在其中记录detect的运行细节

	// ***生成模版 ***
	//generate prototype [1,2][A,B,C,D]
	//-r: input, the half size of prototype  ( R=2*r+1 )
	//-Mat*: output, Mat[2][4];
	void generatePrototype(int r, Mat prototype[2][4]);

	// ***生成响应图 ***
	//generate response map [1,2][A,B,C,D]
	//-r: input, the half size of prototype  ( R=2*r+1 )
	//-Mat*: output, Mat[2][4];
	void generateReponseMap(Mat img, Mat prototype[2][4], Mat scores[2][4]);

	// ***生成模版 ***
	//generate score map 
	//-r: input, the half size of prototype  ( R=2*r+1 )
	//-Mat*: output, Mat[2][4];
	void generateScoreMap(Mat scores[2][4],  Mat& score);

	//***非极大值抑制***
	//Non Maximum Suppression
	//当r很大时非常缓慢，因为使用了膨胀
	//(逐候选点进行判断, 写并行会很快，咕咕~)
	//when r is larger, the speed will be slow because of the operation of dilate
	//-scoreList: output,  vector of [x,y,score]
	//-r: input, radius of NMS
	//-t: input, threshold of NMS, the value less than t will be ignore 
	void NMS(Mat img, vector<Vec3d>& scoreList, int r, int t = 0);

	//***合并矩阵***
	//merge Mat
	//-src: input, should be same size,should be CV_64FC1
	//-dst: output, same size and type as input
	//-flag:  0(merge with min value); 1(merge with max value)
	void merge(Mat src0, Mat src1, Mat & dst,int flag);
	void merge(Mat src0, Mat src1, Mat src2, Mat src3, Mat& dst, int flag);
	
	
};

