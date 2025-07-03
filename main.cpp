#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "nets/nn.h"

using namespace cv;
using namespace std;
using namespace Ort;

int main()
{
	TextDetector detect_model;
	TextClassifier angle_model;
	TextRecognizer rec_model;

	string imgpath = "assets/sample-id.png";
	Mat srcimg = imread(imgpath);
	
	if (srcimg.empty()) {
		cout << "Could not open or find the image: " << imgpath << endl;
		return -1;
	}

	vector<vector<Point2f>> results = detect_model.detect(srcimg);

	for (size_t i = 0; i < results.size(); i++)
	{
		Mat textimg = detect_model.get_rotate_crop_image(srcimg, results[i]);
		
		if (textimg.empty() || textimg.rows == 0 || textimg.cols == 0) {
			cout << "Warning: Empty or invalid text image for region " << i << endl;
			continue;
		}
		
		if (angle_model.predict(textimg) == 1)
		{
			rotate(textimg, textimg, ROTATE_180);
		}
		
		string text = rec_model.predict_text(textimg);
		cout << "Text " << i << ": " << text << endl;
	}
	
	detect_model.draw_pred(srcimg, results);
	
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
	
	return 0;
}