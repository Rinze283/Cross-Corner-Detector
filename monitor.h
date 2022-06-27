#pragma once
#include "config.h"
#include "cornerDetector.h"

using namespace cv;
using namespace std;

class monitor
{
public:

	monitor();

	//***չʾϸ��***
	//show details
	//��ʾʹ��ģ�漰���Ӧ����Ӧͼ���Լ����ĵ÷�ͼ
	//show the prototype and it's response map, final score map
	void showDetails(const cornerDetector::record& record);

	//***��ʾ***
	//show
	//��ʾ���ѵ�ʱ�䣬�Լ����Ľǵ�÷�(ɫ��Խů�÷�Խ��)
	//show cost time, the final corner score(warmer color means higher score)
	void show(const string winName,const Mat& img, const cornerDetector::record& record);

private:

	Mat color;
	static const int shift = 5;

	static Mat3b get_imgShow(const Mat& img);
	static void put_runtime(Mat3b& img, const vector<String>& name, const vector<double>& times);
};

