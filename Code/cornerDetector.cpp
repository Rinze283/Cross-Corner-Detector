#include "cornerDetector.h"

void cornerDetector::detecorCorner(Mat img, vector<Vec3d>& scoreList, int r_pro,int r_nms, int t, record* addr_record)
{
	//记录信息
	//record
	this->p_record = addr_record;
	if (addr_record == NULL)	this->is_record = false;
	else
	{
		this->is_record = true;
		(*this->p_record).reset();
	}

	TIME_START(t_detect);

	//生成模板
	//generate prototype
	Mat prototype[2][4];
	generatePrototype(r_pro, prototype);

	//计算响应图
	//compute response img	
	Mat response[2][4];//A1,B1,C1,D1;A2,B2,B3,B4;
	generateReponseMap(img,prototype,response);

	//计算得分图
	//compute score img	
	Mat score;
	generateScoreMap(response, score);

	//非极大值抑制
	//Non Maximum Suppression
	NMS(score, scoreList, r_nms, t);

	TIME_END(t_detect);
	RECORD(scoreList);
}

void cornerDetector::generatePrototype(int r, Mat prototype[2][4])
{
	TIME_START(t_gen_pro);
	assert(r > 0);
	if (this->r == r)
	{
		for (int m = 0; m < 2; m++)
		{
			for (int n = 0; n < 4; n++)
			{
				prototype[m][n] = this->prototype[m][n];
				RECORD(prototype[m][n]);
			}
		}		
		TIME_END(t_gen_pro);
		return;
	}

	int R = 2 * r + 1;

	//生成二维高斯核图案
	//generate the 2D gaussian kernel pattern
	Mat k_1D = getGaussianKernel(R, -1);
	Mat k_2D = k_1D * k_1D.t();

	double scale[2];
	scale[0] = 1 / k_2D.at<double>(r - 1, r - 1);
	scale[1] = 1 / k_2D.at<double>(r - 1, r);

	Rect rect_ABCD[4];
	rect_ABCD[0] = Rect(r + 1, 0, r, r);
	rect_ABCD[1] = Rect(0, r + 1, r, r);
	rect_ABCD[2] = Rect(0, 0, r, r);
	rect_ABCD[3] = Rect(r + 1, r + 1, r, r);

	for (int i = 0; i < 4; i++)
	{
		//生成第一类模版
		//generate the first type
		Rect rect = rect_ABCD[i];
		prototype[0][i] = Mat::zeros(R, R, CV_64FC1);
		k_2D(rect).copyTo(prototype[0][i](rect));
		prototype[0][i] = prototype[0][i] * scale[0];

		//record
		this->prototype[0][i] = prototype[0][i];
		RECORD(prototype[0][i]);

		//生成第二类模版
		//generate the second type
		//A:top B:bottom C:right D:left
		prototype[1][i] = Mat::zeros(R, R, CV_64FC1);
		switch (i)
		{
		case 0:
			for (int m = 0; m < r; m++)
			{
				rect = Rect(m + 1, m, R - 2 - 2 * m, 1);
				k_2D(rect).copyTo(prototype[1][i](rect));
			}
			break;
		case 1:
			for (int m = 0; m < r; m++)
			{
				rect = Rect(m + 1, R - 1 - m, R - 2 - 2 * m, 1);
				k_2D(rect).copyTo(prototype[1][i](rect));
			}
			break;
		case 2:
			for (int m = 0; m < r; m++)
			{
				rect = Rect(R - 1 - m, m + 1, 1, R - 2 - 2 * m);
				k_2D(rect).copyTo(prototype[1][i](rect));
			}
			break;
		case 3:
			for (int m = 0; m < r; m++)
			{
				rect = Rect(m, m + 1, 1, R - 2 - 2 * m);
				k_2D(rect).copyTo(prototype[1][i](rect));
			}
			break;
		}
		prototype[1][i] = prototype[1][i] * scale[1];

		//record
		this->prototype[1][i] = prototype[1][i];
		RECORD(prototype[1][i]);
	}
	//更新r
	//update r
	this->r = r;

	TIME_END(t_gen_pro);
}

void cornerDetector::generateReponseMap(cv::Mat img, Mat prototype[2][4], cv::Mat response[2][4])
{
	TIME_START(t_gen_res);

	for (size_t m = 0; m < 2; m++)
	{
		for (size_t n = 0; n < 4; n++)
		{
			response[m][n] = Mat::zeros(img.size(), CV_64F);
			filter2D(img, response[m][n], CV_64F, prototype[m][n]);
			RECORD(response[m][n]);
		}
	}

	TIME_END(t_gen_res);
}

void cornerDetector::generateScoreMap(cv::Mat  scores[2][4], cv::Mat& score)
{
	TIME_START(t_gen_sco);

	Mat meanScore[2];
	for (size_t i = 0; i < 2; i++)
	{
		meanScore[i] = (scores[i][0] + scores[i][1] + scores[i][2] + scores[i][3]) / 4;
	}

	Mat minAB[2], minCD[2], minS1[2], minS2[2];
	for (int i = 0; i < 2; i++)
	{
		merge(scores[i][0], scores[i][1], minAB[i], 0);
		merge(scores[i][2], scores[i][3], minCD[i], 0);
		merge(minAB[i] - meanScore[i], meanScore[i] - minCD[i], minS1[i], 0);
		merge(meanScore[i] - minAB[i], minCD[i] - meanScore[i], minS2[i], 0);
	}
	merge(minS1[0], minS2[0], minS1[1], minS2[1], score, 1);

	RECORD(score);
	TIME_END(t_gen_sco);
}

void cornerDetector::NMS(Mat img, vector<Vec3d>& scoreList, int r, int t)
{
	assert(img.type() == 6);
	assert(r >= 3 && t >= 0);
	assert(scoreList.empty());

	TIME_START(t_nms);

	Mat srcDilate, dstlogic;
	Mat element = getStructuringElement(MORPH_RECT, Size(r, r));
	dilate(img, srcDilate, element);
	dstlogic = img == srcDilate & img >= t;

	vector<Vec2i> ptList;
	findNonZero(dstlogic, ptList);

	for (const Vec2i& v : ptList)
	{
		double score = img.at<double>(v[1], v[0]);
		scoreList.push_back(Vec3d(v[0], v[1], score));

		RECORD_LARGER_TO(maxScore, score);
		RECORD_SMALLER_TO(minScore, score);
	}

	TIME_END(t_nms);
}

void cornerDetector::merge(Mat src0, Mat src1, Mat& dst, int flag)
{
	assert(flag == 0 || flag == 1);
	assert(!src0.empty()&& !src1.empty() && dst.empty());
	assert(src0.size() == src1.size());
	assert(src0.type() == src1.type());
	
	dst = Mat(src0.size(), src0.type());
	int nRows = src0.rows;
	int nCols = src0.cols;
	if (src0.isContinuous()&&src1.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}
	int i, j;
	double* p0,* p1,* p;
	if (flag == 0)
	{
		for (i = 0; i < nRows; ++i)
		{
			p = dst.ptr<double>(i);
			p0 = src0.ptr<double>(i);
			p1 = src1.ptr<double>(i);
			for (j = 0; j < nCols; ++j)
			{
				p[j] = p0[j] < p1[j] ? p0[j] : p1[j];
			}
		}
	}
	else if (flag == 1)
	{
		for (i = 0; i < nRows; ++i)
		{
			p = dst.ptr<double>(i);
			p0 = src0.ptr<double>(i);
			p1 = src1.ptr<double>(i);
			for (j = 0; j < nCols; ++j)
			{
				p[j] = p0[j] > p1[j] ? p0[j] : p1[j];
			}
		}
	}
	else
		return;

}

void cornerDetector::merge(Mat src0, Mat src1, Mat src2, Mat src3, Mat& dst, int flag)
{
	assert(flag == 0 || flag == 1);
	assert(!src0.empty() && !src1.empty() && !src2.empty() && !src3.empty() && dst.empty());
	Size s = src0.size();
	assert(src1.size() == s&&src2.size()==s &&src3.size() == s);
	assert(src0.type() == 6&&src1.type() == 6&&src2.type() == 6&&src3.type() == 6);

	dst = Mat(src0.size(), src0.type());
	int nRows = src0.rows;
	int nCols = src0.cols;
	if (src0.isContinuous()&& src1.isContinuous()&& src2.isContinuous()&& src3.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}
	int i, j;
	double* p0, * p1, *p2,*p3,* p;
	if (flag == 0)
	{
		for (i = 0; i < nRows; ++i)
		{
			p = dst.ptr<double>(i);
			p0 = src0.ptr<double>(i);
			p1 = src1.ptr<double>(i);
			p2 = src2.ptr<double>(i);
			p3 = src3.ptr<double>(i);
			for (j = 0; j < nCols; ++j)
			{
				p[j] = p0[j] < p1[j] ? p0[j] : p1[j];
				p[j] = p[j] < p2[j] ? p[j] : p2[j];
				p[j] = p[j] < p3[j] ? p[j] : p3[j];
			}
		}
	}
	else if (flag == 1)
	{
		for (i = 0; i < nRows; ++i)
		{
			p = dst.ptr<double>(i);
			p0 = src0.ptr<double>(i);
			p1 = src1.ptr<double>(i);
			p2 = src2.ptr<double>(i);
			p3 = src3.ptr<double>(i);
			for (j = 0; j < nCols; ++j)
			{
				p[j] = p0[j] > p1[j] ? p0[j] : p1[j];
				p[j] = p[j] > p2[j] ? p[j] : p2[j];
				p[j] = p[j] > p3[j] ? p[j] : p3[j];
			}
		}
	}
	else
		return;
}


