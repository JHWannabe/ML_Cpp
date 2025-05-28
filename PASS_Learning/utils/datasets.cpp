// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <png.h>
#include "../png++/png.hpp"
// For Original Header
#include "transforms.hpp"
#include "datasets.hpp"

namespace fs = std::filesystem;

// -----------------------------------------------
// namespace{datasets} -> function{collect}
// -----------------------------------------------
void datasets::collect(const std::string root, const std::string sub, std::vector<std::string>& paths, std::vector<std::string>& fnames) {
	try {
		fs::path ROOT(root);
		for (auto& p : fs::directory_iterator(ROOT)) {
			if (!fs::is_directory(p)) {
				std::stringstream rpath, fname;
				rpath << p.path().string();
				fname << p.path().filename().string();
				paths.push_back(rpath.str());
				fnames.push_back(sub + fname.str());
			}
			else {
				std::stringstream subsub;
				subsub << p.path().filename().string();
				datasets::collect(root + '/' + subsub.str(), sub + subsub.str() + '/', paths, fnames);
			}
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error in collect: " << e.what() << std::endl;
	}
}

// -----------------------------------------------
// namespace{datasets} -> function{LoadImageFromFile}
// -----------------------------------------------
cv::Mat datasets::LoadImageFromFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
		return cv::Mat();
	}

	return cv::imread(filename, cv::IMREAD_COLOR);
}

// -----------------------------------------------
// namespace{datasets} -> function{RGB_Loader}
// -----------------------------------------------
cv::Mat datasets::RGB_Loader(std::string& path) {
	cv::Mat BGR;
	BGR = LoadImageFromFile(path);

	if (BGR.empty()) {
		std::cerr << "Error : Couldn't open the image '" << path << "'." << std::endl;
		std::exit(1);
	}

	return BGR;
}

// -----------------------------------------------
// namespace{datasets} -> function{GRAY_Loader}
// -----------------------------------------------
cv::Mat datasets::GRAY_Loader(std::string& path) {
	cv::Mat BGR, GRAY;
	BGR = LoadImageFromFile(path);

	if (BGR.empty()) {
		std::cerr << "Error : Couldn't open the image '" << path << "'." << std::endl;
		std::exit(1);
	}
	cv::cvtColor(BGR, GRAY, cv::COLOR_BGR2GRAY);
	return GRAY.clone();
}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{SegmentImageWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::SegmentImageWithPaths::SegmentImageWithPaths(const std::string root, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform, const std::string mode, cv::Size resize) {
	datasets::collect(root, "", this->paths, this->fnames);
	std::sort(this->paths.begin(), this->paths.end());
	std::sort(this->fnames.begin(), this->fnames.end());
	this->imageTransform = imageTransform;
	this->labelTransform = labelTransform;
	this->mode = mode;
	this->resize = resize;
}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{SegmentImageWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::SegmentImageWithPaths::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, int, cv::Mat, cv::Mat>& data) {
	std::string file_path = this->paths.at(idx);
	std::string fname = this->fnames.at(idx);
	cv::Mat img, label;
	std::tuple<cv::Mat> augmentImgs;

	img = datasets::RGB_Loader(file_path);
	label = cv::Mat::zeros(this->resize, CV_8UC1);

	if (this->mode == "super") {
		std::string label_path = file_path;
		if (file_path.find("notfound") != std::string::npos) {
			size_t pos;
			if ((pos = label_path.find("bmp")) != std::string::npos) { label_path.replace(pos, 3, "jpg"); }
			if ((pos = label_path.find("origin")) != std::string::npos) { label_path.replace(pos, 6, "label"); }
			label = datasets::GRAY_Loader(label_path);
		}
	}
	else if (this->mode == "unsuper") {
		Augmentation augmentor;
		std::tie(img, label) = augmentor.generateAnomaly(file_path, this->resize);
	}
	else if (this->mode == "test" || this->mode == "valid") {
		if (this->paths.at(idx).find("NG") != std::string::npos) { this->y_true = 0; }
		if (this->paths.at(idx).find("notfound") != std::string::npos) { this->y_true = 0; }

		std::string label_path = this->paths.at(idx);
		size_t pos;
		if ((pos = label_path.find("bmp")) != std::string::npos) { label_path.replace(pos, 4, "jpg"); }
		if ((pos = label_path.find("origin")) != std::string::npos) { label_path.replace(pos, 6, "label"); }
		label = datasets::GRAY_Loader(label_path);
	}

	cv::resize(img, img, this->resize);
	cv::resize(label, label, this->resize);

	torch::Tensor image_tensor = transforms::apply(this->imageTransform, img);
	torch::Tensor label_tensor = transforms::apply(this->labelTransform, label).squeeze(0);

	// 1) float 타입으로 변환
	label_tensor = label_tensor.to(torch::kFloat32);
	// 2) 현재 최대값 추출
	float max_val = label_tensor.max().item<float>();
	// 3) (max_val > 0)일 때만 정규화
	if (max_val > 0.0f) {
		label_tensor = label_tensor / max_val;
	}

	data = { image_tensor.detach().clone(), label_tensor.detach().clone(), fname, this->y_true, img, label };

	return;
}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{SegmentImageWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::SegmentImageWithPaths::size() {
	return this->fnames.size();
}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{Augmentation} -> function{generateAnomaly}
// -------------------------------------------------------------------------
std::tuple<cv::Mat, cv::Mat> datasets::Augmentation::generateAnomaly(std::string& file_path, cv::Size resize)
{
	// (0) p 값을 0~1 사이 난수로 생성
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> p_dist(0.0f, 1.0f);
	float p = p_dist(gen);

	cv::Mat img = datasets::RGB_Loader(file_path);
	cv::resize(img, img, resize);
	cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
	
	if (p > 0.5) {
		// (1) Generate Mask		
		cv::Mat mask_f, mask = generatePerlinNoise(img);
		cv::cvtColor(mask, mask_f, cv::COLOR_GRAY2BGR);

		// (2) Generate Stable Diffusion Image
		cv::Mat stableDiffusionImg = stableDiffusion(file_path);
		cv::resize(stableDiffusionImg, stableDiffusionImg, resize);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dist(0.15f, 1.0f);
		float factor = dist(gen);
		cv::Mat stable_f, img_f;
		stableDiffusionImg.convertTo(stable_f, CV_32FC3);
		img.convertTo(img_f, CV_32FC3);
		cv::Mat anomalySourceImg = factor * mask_f.mul(stable_f) + (1.0f - factor) * (mask_f.mul(img_f));
		cv::cvtColor(anomalySourceImg, anomalySourceImg, cv::COLOR_BGR2RGB);

		// (3) Blend Image and Anomaly Source
		cv::Mat temp = (cv::Scalar(1.0f, 1.0f, 1.0f) - mask_f).mul(img_f);
		anomalySourceImg = temp + anomalySourceImg;
		img = anomalySourceImg;
		img.convertTo(img, CV_8UC3);

		return std::make_tuple(img, mask*255);
	}
	else {
		cv::Mat structureSourceImg = rand_augment(img);

		assert(resize_w % this->grid_size == 0 && resize_h % this->grid_size == 0);

		int gw = structureSourceImg.cols / this->grid_size;
		int gh = structureSourceImg.rows / this->grid_size;
		int grid_count = this->grid_size * this->grid_size;

		// 2. Split into grid blocks
		std::vector<cv::Mat> blocks(grid_count);
		int idx = 0;
		for (int y = 0; y < structureSourceImg.rows; y += gh) {
			for (int x = 0; x < structureSourceImg.cols; x += gw) {
				cv::Rect roi(x, y, gw, gh);
				blocks[idx++] = img(roi).clone();  // 안전하게 clone
			}
		}

		// 3. Shuffle blocks
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(blocks.begin(), blocks.end(), g);

		// 4. Reconstruct image from shuffled blocks
		cv::Mat result(structureSourceImg.rows, structureSourceImg.cols, img.type());
		idx = 0;
		for (int y = 0; y < structureSourceImg.rows; y += gh) {
			for (int x = 0; x < structureSourceImg.cols; x += gw) {
				cv::Mat roi = result(cv::Rect(x, y, gw, gh));
				blocks[idx++].copyTo(roi);
			}
		}

		return std::make_tuple(result, mask);
	}
}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{Augmentation} -> function{generatePerlinNoise}
// -------------------------------------------------------------------------
cv::Mat datasets::Augmentation::generatePerlinNoise(cv::Mat& img)
{
	int min_scale = 0, max_scale = 6;
	std::mt19937 rng(std::random_device{}());
	std::uniform_int_distribution<int> dist1(min_scale, max_scale - 1);

	int scale_x = std::pow(2, dist1(rng));
	int scale_y = std::pow(2, dist1(rng));

	// Perlin 노이즈 생성
	cv::Mat noise = generatePerlinNoise2D(img.cols, img.rows, scale_x, scale_y);

	// Affine 회전 (선택)
	std::uniform_real_distribution<float> dist2(-90.0f, 90.0f);
	float angle = dist2(rng);

	// 2. 회전 중심 = 이미지 중앙
	cv::Point2f center(noise.rows / 2.0f, noise.cols / 2.0f);

	// 3. 회전 매트릭스 계산
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

	// 4. 회전 적용 (경계값은 0)
	cv::Mat dst;
	cv::warpAffine(noise, dst, rot_mat, noise.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

	// Threshold 적용 → float mask
	cv::Mat mask_float, mask;
	float threshold = 0.85f;
	cv::threshold(noise, mask_float, threshold, 1.0, cv::THRESH_BINARY);

	// float(0.0, 1.0) → uint8(0, 1)
	mask_float.convertTo(mask, CV_32FC1);

	return mask;  // CV_8UC1, 값은 0 또는 1
}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{Augmentation} -> function{generatePerlinNoise2D}
// -------------------------------------------------------------------------
cv::Mat datasets::Augmentation::generatePerlinNoise2D(int width, int height, int res_x, int res_y) {
	int dx = width / res_x;
	int dy = height / res_y;
	cv::Mat noise = cv::Mat::zeros(height, width, CV_32F);

	std::vector<std::vector<cv::Point2f>> gradients(res_x + 1, std::vector<cv::Point2f>(res_y + 1));

	// 1. 랜덤 유닛 벡터 생성 (그라디언트)
	std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<float> dist(0, 2 * CV_PI);

	for (int i = 0; i <= res_x; ++i) {
		for (int j = 0; j <= res_y; ++j) {
			float angle = dist(rng);
			gradients[i][j] = cv::Point2f(std::cos(angle), std::sin(angle));
		}
	}

	// 2. 노이즈 생성
	for (int y = 0; y < height; ++y) {
		int gy = y / dy;
		float fy = (float)(y % dy) / dy;
		float vy = 6 * std::pow(fy, 5) - 15 * std::pow(fy, 4) + 10 * std::pow(fy, 3);

		for (int x = 0; x < width; ++x) {
			int gx = x / dx;
			float fx = (float)(x % dx) / dx;
			float vx = 6 * std::pow(fx, 5) - 15 * std::pow(fx, 4) + 10 * std::pow(fx, 3);

			// 코너 그라디언트
			cv::Point2f g00 = gradients[gx][gy];
			cv::Point2f g10 = gradients[gx + 1][gy];
			cv::Point2f g01 = gradients[gx][gy + 1];
			cv::Point2f g11 = gradients[gx + 1][gy + 1];

			// 벡터
			cv::Point2f d00(fx, fy);
			cv::Point2f d10(fx - 1, fy);
			cv::Point2f d01(fx, fy - 1);
			cv::Point2f d11(fx - 1, fy - 1);

			// 내적
			float s = g00.dot(d00);
			float t = g10.dot(d10);
			float u = g01.dot(d01);
			float v = g11.dot(d11);

			float st = s + vx * (t - s);
			float uv = u + vx * (v - u);
			float val = st + vy * (uv - st);
			noise.at<float>(y, x) = val;
		}
	}

	// 정규화
	cv::normalize(noise, noise, 0, 1, cv::NORM_MINMAX);
	return noise;
}

cv::Mat datasets::Augmentation::rand_augment(cv::Mat& img) {
	std::vector<std::function<cv::Mat(const cv::Mat&)>> augmenters;

	// (1) GammaContrast
	augmenters.push_back([](const cv::Mat& input) {
		cv::Mat img; input.convertTo(img, CV_32F, 1.0 / 255.0);
		float gamma = static_cast<float>(rand()) / RAND_MAX * 1.5f + 0.5f;
		cv::pow(img, gamma, img);
		img *= 255.0f;
		img.convertTo(img, CV_8U);
		return img;
		});

	// (2) MultiplyAndAddToBrightness
	augmenters.push_back([](const cv::Mat& input) {
		cv::Mat img;
		float mul = 0.8f + static_cast<float>(rand()) / RAND_MAX * 0.4f;
		int add = rand() % 61 - 30;
		input.convertTo(img, -1, mul, add);
		return img;
		});

	// (3) Invert
	augmenters.push_back([](const cv::Mat& input) {
		return cv::Scalar::all(255) - input;
		});

	// (4) Hue and Saturation shift (in HSV)
	augmenters.push_back([](const cv::Mat& input) {
		cv::Mat hsv, out;
		cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
		std::vector<cv::Mat> channels;
		cv::split(hsv, channels);
		int h_shift = rand() % 101 - 50;
		int s_shift = rand() % 101 - 50;
		channels[0] += h_shift;
		channels[1] += s_shift;
		cv::merge(channels, hsv);
		cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
		return out;
		});

	// (5) Rotation
	augmenters.push_back([](const cv::Mat& input) {
		double angle = (rand() % 21 - 10); // -10~10도
		cv::Point2f center(input.cols / 2.0f, input.rows / 2.0f);
		cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::Mat rotated;
		cv::warpAffine(input, rotated, rot, input.size());
		return rotated;
		});

	// (6) Autocontrast & Equalize (approximation)
	augmenters.push_back([](const cv::Mat& input) {
		cv::Mat img_yuv;
		cv::cvtColor(input, img_yuv, cv::COLOR_BGR2YCrCb);
		std::vector<cv::Mat> channels;
		cv::split(img_yuv, channels);
		cv::equalizeHist(channels[0], channels[0]);
		cv::merge(channels, img_yuv);
		cv::Mat result;
		cv::cvtColor(img_yuv, result, cv::COLOR_YCrCb2BGR);
		return result;
		});

	// Shuffle + pick 3 random
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(augmenters.begin(), augmenters.end(), g);

	// Apply 3 random augmentations in sequence
	cv::Mat output = img.clone();
	for (int i = 0; i < 3; ++i) {
		output = augmenters[i](output);
	}

	return output;
}
