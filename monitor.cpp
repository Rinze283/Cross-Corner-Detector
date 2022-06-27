#include "monitor.h"

monitor::monitor()
{
	Mat1b gray(1, 201);
	for (int i = 0; i < 201; i++)	gray.at<uchar>(i) = 201-i;
	applyColorMap(gray, this->color, COLORMAP_HSV);
}

void monitor::showDetails(const cornerDetector::record& record)
{
	destroyAllWindows();
	for (int m = 0; m < 2; m++)
	{
		for (int n = 0; n < 4; n++)
		{
			string winName = "prototype[" + to_string(m) + "][" + to_string(n) + "]";
			namedWindow(winName, 0);
			resizeWindow(winName, 480, 480);
			imshow(winName,record.prototype[m][n]);
			waitKey(500);

			winName = "response[" + to_string(m) + "][" + to_string(n) + "]";
			namedWindow(winName, 0);
			Mat normal;
			normalize(record.response[m][n], normal, 0.0, 1.0, NORM_MINMAX);
			imshow(winName, normal);
			waitKey(500);
		}
	}
	string winName = "score";
	namedWindow(winName, 0);
	resizeWindow(winName, 480, 480);
	Mat normal;
	normalize(record.score, normal, 0,1,NORM_MINMAX);
	imshow(winName, normal);
	cout << "press keyboard to skip the cv::waitKey(0)" << endl;
	waitKey(0);	
}

void monitor::show(const string winName, const Mat& img, const cornerDetector::record& record)
{
	//cout << winName << endl;
	Mat3b imgShow = get_imgShow(img);
	vector<double> times{ record.t_detect, record.t_gen_pro, record.t_gen_res, record.t_gen_sco,record.t_nms };
	vector<String> name{ "detection", "generate prototype", "generate response map", 
		"generate score map","non maximum suppression" };
	put_runtime(imgShow, name, times);

	// points
	float scale = (float)imgShow.rows / 480;
	for (Vec3d v : record.scoreList)
	{
		Point2d p = Point2d(v[0], v[1]);
		Point2d pt = Point2d(v[0]+2, v[1]-2);
		int rate = 200 * (v[2] - record.minScore) / (record.maxScore - record.minScore);
		Scalar c = this->color.at<Vec3b>(rate);
		circle(imgShow, pow(2, monitor::shift) * p, pow(2, monitor::shift) * scale * 3, c, scale * 1, LINE_AA, monitor::shift);
		//putText(imgShow, to_string((int)v[2]), pt, FONT_ITALIC, scale * 0.3, Scalar(255,255,255), scale * 0.8, LINE_AA);
	}

	namedWindow(winName, WINDOW_NORMAL);
	imshow(winName, imgShow);
}

Mat3b monitor::get_imgShow(const Mat& img)
{
	assert(!img.empty());
	assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

	Mat3b imgShow;
	if (img.type() == CV_8UC1)		cvtColor(img, imgShow, COLOR_GRAY2RGB);
	else							img.copyTo(imgShow);

	return imgShow;
}

void monitor::put_runtime(Mat3b& img, const vector<String>& name, const vector<double>& times)
{
	assert(name.size() == times.size());
	
	int num = name.size();
	float scale = (float)img.rows / 480;

	ostringstream s;
	for (int i = 0; i < num; i++)
	{
		s.str("");	s.clear();
		s << setprecision(1) << fixed << times[i];
		putText(img, name[i] + ":" + s.str() + "ms", scale * Point(5, (i + 1) * 15), FONT_ITALIC, 0.5 * scale, Scalar(255, 255, 0), scale, LINE_AA);
		//cout << name[i] + ":" + s.str() + "ms" << endl;
	}	
}
