#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class TextDetector
{
public:
	TextDetector();
	vector<vector<Point2f>> detect(Mat& srcimg);
	void draw_pred(Mat& srcimg, vector<vector<Point2f>> results);
	Mat get_rotate_crop_image(const Mat& frame, vector<Point2f> vertices);
private:
	float binaryThreshold;
	float polygonThreshold;
	float unclipRatio;
	int maxCandidates;
	const int longSideThresh = 10;
	const int short_size = 960;
	const float meanValues[3] = { 0.485, 0.456, 0.406 };
	const float normValues[3] = { 0.229, 0.224, 0.225 };
	float contourScore(const Mat& binary, const vector<Point>& contour);
	void unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly);
	Mat preprocess(Mat srcimg);
	vector<float> input_image_;
	void normalize_(Mat img);

	Session *net;
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "DBNet");
	SessionOptions sessionOptions = SessionOptions();
	vector<Ort::AllocatedStringPtr> input_name_ptrs;
	vector<Ort::AllocatedStringPtr> output_name_ptrs;
	vector<const char*> input_names;
	vector<const char*> output_names;
};

class TextClassifier
{
public:
	TextClassifier();
	int predict(Mat cv_image);
private:
	const int label_list[2] = { 0, 180 };

	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	const int inpWidth = 192;
	const int inpHeight = 48;
	int num_out;
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Angle classify");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<Ort::AllocatedStringPtr> input_name_ptrs;
	vector<Ort::AllocatedStringPtr> output_name_ptrs;
	vector<const char*> input_names;
	vector<const char*> output_names;
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;
};

class TextRecognizer
{
public:
	TextRecognizer();
	string predict_text(Mat cv_image);
	
private:
	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	const int inpWidth = 320;
	const int inpHeight = 48;
	
	vector<float> input_image_;
	vector<string> alphabet;
	int names_len;
	vector<int> preb_label;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "CRNN");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<Ort::AllocatedStringPtr> input_name_ptrs;
	vector<Ort::AllocatedStringPtr> output_name_ptrs;
	vector<const char*> input_names;
	vector<const char*> output_names;
	vector<vector<int64_t>> input_node_dims;
	vector<vector<int64_t>> output_node_dims;
};