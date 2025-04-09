#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <string>
#include <tuple>
#include <vector>
#include <random>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// For Original Header
#include "transforms.hpp"
#include "fastNoiseLite.h"

// -----------------------
// namespace{datasets}
// -----------------------
namespace datasets {

    // Function Prototype
    void collect(const std::string root, const std::string sub, std::vector<std::string>& paths, std::vector<std::string>& fnames);
    cv::Mat RGB_Loader(std::string& path);
    cv::Mat GRAY_Loader(std::string& path);
    cv::Mat LoadImageFromFile(const std::string& filename);

    // ----------------------------------------------------
    // namespace{datasets} -> class{Augmentation}
    // ----------------------------------------------------
    class Augmentation {
    public:
        std::tuple<cv::Mat, cv::Mat> generateAnomaly(cv::Mat& img);
        cv::Mat generatePerlinNoise(cv::Mat& img);
		cv::Mat stableDiffusion(cv::Mat& img);
        cv::Mat generatePerlinNoise2D(int width, int height, int res_x, int res_y);
    };

    // ----------------------------------------------------
    // namespace{datasets} -> class{SegmentImageWithPaths}
    // ----------------------------------------------------
    class SegmentImageWithPaths {
    private:
        int y_true = 1;
        std::string mode;
        std::vector<transforms_Compose> imageTransform, labelTransform;
        std::vector<std::string> paths, fnames;
        cv::Size resize;
    public:
        SegmentImageWithPaths() {}
        SegmentImageWithPaths(const std::string root, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform, const std::string mode, cv::Size resize);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, int, cv::Mat, cv::Mat>& data);
        size_t size();
    };
}

#endif