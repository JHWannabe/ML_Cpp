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

// -----------------------
// namespace{datasets}
// -----------------------
namespace datasets {

    // Function Prototype
    void collect(const std::string root, const std::string sub, std::vector<std::string>& paths, std::vector<std::string>& fnames);
    torch::Tensor Data1d_Loader(std::string& path);
    cv::Mat RGB_Loader(std::string& path);
    cv::Mat Index_Loader(std::string& path);
    std::tuple<torch::Tensor, torch::Tensor> BoundingBox_Loader(std::string& path);

    // ----------------------------------------------------
    // namespace{datasets} -> class{SegmentImageWithPaths}
    // ----------------------------------------------------
    class SegmentImageWithPaths {
    private:
        int y_true = 1;
        std::string mode;
        std::vector<transforms_Compose> transform;
        std::vector<std::string> paths, fnames;
    public:
        SegmentImageWithPaths() {}
        SegmentImageWithPaths(const std::string root, std::vector<transforms_Compose>& transform, const std::string mode);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, int>& data);
        size_t size();
    };
}

// ----------------------------------------------------
// class{AnomalyGenerator}
// ----------------------------------------------------
class AnomalyGenerator {
public:
    AnomalyGenerator() {}
    cv::Mat generateTargetForegroundMask(cv::Mat img);
    cv::Mat anomalySource(cv::Mat img);
    std::vector<std::pair<cv::Mat, cv::Mat>> generateAnomaly(cv::Mat img);
    cv::Mat generatePerlinNoiseMask(cv::Mat img);
    cv::Mat rand_perlin_2d_np(cv::Size shape, cv::Size res);

private:
    cv::Size resize;
    float threshold = 0.5;
    int maxScale = 6, minScale = 0;
    std::pair<float, float> transparency_range;float lerp(float a, float b, float t) { return a + t * (b - a); }
    float fade(float t) { return 6 * t * t * t * t * t - 15 * t * t * t * t + 10 * t * t * t; }
};

#endif