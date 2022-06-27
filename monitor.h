#pragma once
#include "config.h"
#include "cornerDetector.h"

using namespace cv;
using namespace std;

class monitor
{
public:

	monitor();

	//***展示细节***
	//show details
	//显示使用模版及其对应的响应图，以及最后的得分图
	//show the prototype and it's response map, final score map
	void showDetails(const cornerDetector::record& record);

	//***显示***
	//show
	//显示花费的时间，以及最后的角点得分(色调越暖得分越高)
	//show cost time, the final corner score(warmer color means higher score)
	void show(const string winName,const Mat& img, const cornerDetector::record& record);

private:

	Mat color;
	static const int shift = 5;

	static Mat3b get_imgShow(const Mat& img);
	static void put_runtime(Mat3b& img, const vector<String>& name, const vector<double>& times);
};

