#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "cornerDetector.h"
#include "monitor.h"
#include <vector>

using namespace cv;
using namespace std;
using std::vector;
vector<Point2i> points;

const int NUM_IMG=3;

int main(int argc, char** argv[])
{
	cout <<endl;
	cout << "///////////////////////////////////////////////////////////////////////" << endl;
	cout << "/////////////////////CheckBoard Corner Detection of////////////////////" << endl;
	cout << "///Automatic Camera and Range Sensor Calibration using a single Shot///" << endl;
	cout << "///////////////////////////////////////////////////////////////////////" << endl;
	cout << endl;

	//*****************************read img***********************************

	Mat src[NUM_IMG];
	monitor mon;
	string imgName;

	for (int i = 0; i < NUM_IMG; i++)
	{
		imgName= to_string(i) + ".bmp";
		src[i] = imread(imgName, 0);
		if (src[i].empty())
		{
			printf("Cannot read image file");
			return -1;
		}
	}

	//*****************************detection***********************************

	cornerDetector cd;
	cornerDetector::record record;

	bool showDetails = false;
	
	for (int r = 3; r <=12; r+=3)// for different radius
	{
		for (int i = 0; i < NUM_IMG; i++)// for different img
		{
			vector<Vec3d> scoreList;
			cd.detecorCorner(src[i], scoreList, r, 20, 10*r*r,&record);
			
			int interval = 50;
			if(showDetails) 
			{
				interval = 0;
				mon.showDetails(record);//show prototype, response map, score map
			}

			string winName = "img[" + to_string(i) + "]:r[" + to_string(r) + "]";				
			mon.show(winName,src[i], record);// show img

			waitKey(interval);                     
		}				
	}
	waitKey(0);

	return 0;
}
