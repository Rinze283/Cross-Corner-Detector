#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/features2d.hpp>
#include <iomanip>
#include <map>
#include <fstream>

using namespace cv;
using namespace std;


#define TIME_START(X)	if (this->is_record)	(*this->p_record).X = (double)getTickCount();
#define TIME_END(X)		if (this->is_record)	(*this->p_record).X = 1000 * ((double)getTickCount() - (*this->p_record).X) / getTickFrequency();
#define RECORD(X)		if (this->is_record)	(*this->p_record).X = X;
#define RECORD_TO(X,Y)		if (this->is_record)	(*this->p_record).X = Y;
#define RECORD_LARGER_TO(X,Y)		if (this->is_record)	(*this->p_record).X = (*this->p_record).X>Y?(*this->p_record).X:Y;
#define RECORD_SMALLER_TO(X,Y)		if (this->is_record)	(*this->p_record).X = (*this->p_record).X<Y?(*this->p_record).X:Y;
#define TRY_THROW(X,Y)	if (!X)	throw Y;
