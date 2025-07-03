#include "nn.h"

TextDetector::TextDetector()
{
	this->binaryThreshold = 0.7;
	this->polygonThreshold = 0.7;
	this->unclipRatio = 1.6;
	this->maxCandidates = 1000;

	string model_path = "weights/ch_PP-OCRv3_det_infer.onnx";
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	net = new Session(env, model_path.c_str(), sessionOptions);
	
	size_t numInputNodes = net->GetInputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		auto input_name = net->GetInputNameAllocated(i, allocator);
		input_name_ptrs.push_back(std::move(input_name));
		input_names.push_back(input_name_ptrs.back().get());
	}
	
	size_t numOutputNodes = net->GetOutputCount();
	for (int i = 0; i < numOutputNodes; i++)
	{
		auto output_name = net->GetOutputNameAllocated(i, allocator);
		output_name_ptrs.push_back(std::move(output_name));
		output_names.push_back(output_name_ptrs.back().get());
	}
}

Mat TextDetector::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(960, 960), INTER_LINEAR);
	return dstimg;
}

void TextDetector::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - this->meanValues[c]) / this->normValues[c];
			}
		}
	}
}

vector<vector<Point2f>> TextDetector::detect(Mat& srcimg)
{
	int h = srcimg.rows;
	int w = srcimg.cols;
	Mat dstimg = this->preprocess(srcimg);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, dstimg.rows, dstimg.cols };
	
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = net->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());
	const float* floatArray = ort_outputs[0].GetTensorMutableData<float>();
	int outputCount = 1;
	for(int i=0; i < ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().size(); i++)
	{
		int dim = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(i);
		outputCount *= dim;
	}

	Mat binary(dstimg.rows, dstimg.cols, CV_32FC1);
	memcpy(binary.data, floatArray, outputCount * sizeof(float));

	Mat bitmap;
	threshold(binary, bitmap, binaryThreshold, 255, THRESH_BINARY);
	float scaleHeight = (float)(h) / (float)(binary.size[0]);
	float scaleWidth = (float)(w) / (float)(binary.size[1]);
	vector<vector<Point>> contours;
	bitmap.convertTo(bitmap, CV_8UC1);
	findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	size_t numCandidate = min(contours.size(), (size_t)(maxCandidates > 0 ? maxCandidates : INT_MAX));
	vector<float> confidences;
	vector<vector<Point2f>> results;
	for (size_t i = 0; i < numCandidate; i++)
	{
		vector<Point>& contour = contours[i];

		if (contourScore(binary, contour) < polygonThreshold)
			continue;
		
		double area = contourArea(contour);
		if (area < 100)
			continue;   

		vector<Point> contourScaled; contourScaled.reserve(contour.size());
		for (size_t j = 0; j < contour.size(); j++)
		{
			contourScaled.push_back(Point(int(contour[j].x * scaleWidth),
				int(contour[j].y * scaleHeight)));
		}

		RotatedRect box = minAreaRect(contourScaled);
		float longSide = std::max(box.size.width, box.size.height);
		if (longSide < longSideThresh) 
		{
			continue;
		}

		const float angle_threshold = 60;
		bool swap_size = false;
		if (box.size.width < box.size.height)
			swap_size = true;
		else if (fabs(box.angle) >= angle_threshold)
			swap_size = true;
		if (swap_size)
		{
			swap(box.size.width, box.size.height);
			if (box.angle < 0)
				box.angle += 90;
			else if (box.angle > 0)
				box.angle -= 90;
		}

		Point2f vertex[4];
		box.points(vertex);
		vector<Point2f> approx;
		for (int j = 0; j < 4; j++)
			approx.emplace_back(vertex[j]);
		vector<Point2f> polygon;
		unclip(approx, polygon);

		box = minAreaRect(polygon);
		longSide = std::max(box.size.width, box.size.height);
		if (longSide < longSideThresh+2)
		{
			continue;
		}

		results.push_back(polygon);
	}
	confidences = vector<float>(contours.size(), 1.0f);
	return results;
}

void TextDetector::draw_pred(Mat& srcimg, vector<vector<Point2f>> results)
{
	for (int i = 0; i < results.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			circle(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), 4, Scalar(0, 0, 255), -1);
			if (j < 3)
			{
				line(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), Point((int)results[i][j + 1].x, (int)results[i][j + 1].y), Scalar(0, 0, 255), 3);
			}
			else
			{
				line(srcimg, Point((int)results[i][j].x, (int)results[i][j].y), Point((int)results[i][0].x, (int)results[i][0].y), Scalar(0, 0, 255), 3);
			}
		}
	}
}

float TextDetector::contourScore(const Mat& binary, const vector<Point>& contour)
{
	Rect rect = boundingRect(contour);
	int xmin = max(rect.x, 0);
	int xmax = min(rect.x + rect.width, binary.cols - 1);
	int ymin = max(rect.y, 0);
	int ymax = min(rect.y + rect.height, binary.rows - 1);

	Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

	Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
	vector<Point> roiContour;
	for (size_t i = 0; i < contour.size(); i++) {
		Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
		roiContour.push_back(pt);
	}
	vector<vector<Point>> roiContours = { roiContour };
	fillPoly(mask, roiContours, Scalar(1));
	float score = mean(binROI, mask).val[0];
	return score;
}

void TextDetector::unclip(const vector<Point2f>& inPoly, vector<Point2f> &outPoly)
{
	float area = contourArea(inPoly);
	float length = arcLength(inPoly, true);
	float distance = area * unclipRatio / length;

	size_t numPoints = inPoly.size();
	vector<vector<Point2f>> newLines;
	for (size_t i = 0; i < numPoints; i++)
	{
		vector<Point2f> newLine;
		Point pt1 = inPoly[i];
		Point pt2 = inPoly[(i - 1) % numPoints];
		Point vec = pt1 - pt2;
		float unclipDis = (float)(distance / norm(vec));
		Point2f rotateVec = Point2f(vec.y * unclipDis, -vec.x * unclipDis);
		newLine.push_back(Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
		newLine.push_back(Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
		newLines.push_back(newLine);
	}

	size_t numLines = newLines.size();
	for (size_t i = 0; i < numLines; i++)
	{
		Point2f a = newLines[i][0];
		Point2f b = newLines[i][1];
		Point2f c = newLines[(i + 1) % numLines][0];
		Point2f d = newLines[(i + 1) % numLines][1];
		Point2f pt;
		Point2f v1 = b - a;
		Point2f v2 = d - c;
		float cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

		if (fabs(cosAngle) > 0.7)
		{
			pt.x = (b.x + c.x) * 0.5;
			pt.y = (b.y + c.y) * 0.5;
		}
		else
		{
			float denom = a.x * (float)(d.y - c.y) + b.x * (float)(c.y - d.y) +
				d.x * (float)(b.y - a.y) + c.x * (float)(a.y - b.y);
			float num = a.x * (float)(d.y - c.y) + c.x * (float)(a.y - d.y) + d.x * (float)(c.y - a.y);
			float s = num / denom;

			pt.x = a.x + s * (b.x - a.x);
			pt.y = a.y + s * (b.y - a.y);
		}
		outPoly.push_back(pt);
	}
}

Mat TextDetector::get_rotate_crop_image(const Mat& frame, vector<Point2f> vertices)
{
	Rect rect = boundingRect(Mat(vertices));
	
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, frame.cols - rect.x);
	rect.height = min(rect.height, frame.rows - rect.y);
	
	if (rect.width <= 0 || rect.height <= 0 || 
		rect.x >= frame.cols || rect.y >= frame.rows) {
		return Mat();
	}
	
	Mat crop_img = frame(rect);

	const Size outputSize = Size(rect.width, rect.height);

	vector<Point2f> targetVertices{ Point2f(0, outputSize.height),Point2f(0, 0), Point2f(outputSize.width, 0), Point2f(outputSize.width, outputSize.height)};

	for (int i = 0; i < 4; i++)
	{
		vertices[i].x -= rect.x;
		vertices[i].y -= rect.y;
	}
	
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(crop_img, result, rotationMatrix, outputSize, cv::BORDER_REPLICATE);
	return result;
}

TextClassifier::TextClassifier()
{
	string model_path = "weights/ch_ppocr_mobile_v2.0_cls_train.onnx";
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		auto input_name = ort_session->GetInputNameAllocated(i, allocator);
		input_name_ptrs.push_back(std::move(input_name));
		input_names.push_back(input_name_ptrs.back().get());
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		auto output_name = ort_session->GetOutputNameAllocated(i, allocator);
		output_name_ptrs.push_back(std::move(output_name));
		output_names.push_back(output_name_ptrs.back().get());
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	num_out = output_node_dims[0][1];
}

Mat TextClassifier::preprocess(Mat srcimg)
{
	Mat dstimg;
	int h = srcimg.rows;
	int w = srcimg.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inpHeight * ratio));
	if (ceil(this->inpHeight*ratio) > this->inpWidth)
	{
		resized_w = this->inpWidth;
	}

	resize(srcimg, dstimg, Size(resized_w, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

void TextClassifier::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(this->inpHeight * this->inpWidth * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < inpWidth; j++)
			{
				if (j < col)
				{
					float pix = img.ptr<uchar>(i)[j * 3 + c];
					this->input_image_[c * row * inpWidth + i * inpWidth + j] = (pix / 255.0 - 0.5) / 0.5;
				}
				else
				{
					this->input_image_[c * row * inpWidth + i * inpWidth + j] = 0;
				}
			}
		}
	}
}

int TextClassifier::predict(Mat cv_image)
{
	Mat dstimg = this->preprocess(cv_image);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	
	int max_id = 0;
	float max_prob = -1;
	for (int i = 0; i < num_out; i++)
	{
		if (pdata[i] > max_prob)
		{
			max_prob = pdata[i];
			max_id = i;
		}
	}

	return max_id;
}

TextRecognizer::TextRecognizer()
{
	string model_path = "weights/korean_PP-OCRv3_rec_infer.onnx";
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		auto input_name = ort_session->GetInputNameAllocated(i, allocator);
		input_name_ptrs.push_back(std::move(input_name));
		input_names.push_back(input_name_ptrs.back().get());
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		auto output_name = ort_session->GetOutputNameAllocated(i, allocator);
		output_name_ptrs.push_back(std::move(output_name));
		output_names.push_back(output_name_ptrs.back().get());
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}

	ifstream ifs("assets/korean_dict.txt");
	string line;
	while (getline(ifs, line))
	{
		this->alphabet.push_back(line);
	}
	this->alphabet.push_back(" ");
	names_len = this->alphabet.size();
}

Mat TextRecognizer::preprocess(Mat srcimg)
{
	Mat dstimg;
	int h = srcimg.rows;
	int w = srcimg.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inpHeight * ratio));
	if (ceil(this->inpHeight*ratio) > this->inpWidth)
	{
		resized_w = this->inpWidth;
	}
	
	resize(srcimg, dstimg, Size(resized_w, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

void TextRecognizer::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(this->inpHeight * this->inpWidth * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < inpWidth; j++)
			{
				if (j < col)
				{
					float pix = img.ptr<uchar>(i)[j * 3 + c];
					this->input_image_[c * row * inpWidth + i * inpWidth + j] = (pix / 255.0 - 0.5) / 0.5;
				}
				else
				{
					this->input_image_[c * row * inpWidth + i * inpWidth + j] = 0;
				}
			}
		}
	}
}

string TextRecognizer::predict_text(Mat cv_image)
{
	Mat dstimg = this->preprocess(cv_image);
	this->normalize_(dstimg);
	
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());
	
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	
	int i = 0, j = 0;
	int h = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(2);
	int w = ort_outputs.at(0).GetTensorTypeAndShapeInfo().GetShape().at(1);
	
	preb_label.resize(w);
	for (i = 0; i < w; i++)
	{
		int one_label_idx = 0;
		float max_data = -10000;
		for (j = 0; j < h; j++)
		{
			float data_ = pdata[i*h + j];
			if (data_ > max_data)
			{
				max_data = data_;
				one_label_idx = j;
			}
		}
		preb_label[i] = one_label_idx;
	}
	
	vector<int> no_repeat_blank_label;
	for (size_t elementIndex = 0; elementIndex < w; ++elementIndex)
	{
		if (preb_label[elementIndex] != 0 && !(elementIndex > 0 && preb_label[elementIndex - 1] == preb_label[elementIndex]))
		{
			int idx = preb_label[elementIndex] - 1;
			if (idx >= 0 && idx < alphabet.size()) {
				no_repeat_blank_label.push_back(idx);
			}
		}
	}
	
	int len_s = no_repeat_blank_label.size();
	string plate_text;
	for (i = 0; i < len_s; i++)
	{
		plate_text += alphabet[no_repeat_blank_label[i]];
	}
	
	return plate_text;
}