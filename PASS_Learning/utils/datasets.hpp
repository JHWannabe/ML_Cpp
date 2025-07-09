#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <tuple>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
// For External Library
#include <torch/torch.h>
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// For Original Header
#include "transforms.hpp"
#include "fastNoiseLite.h"
#include "diffusion.h"

namespace py = pybind11;

// -----------------------
// namespace{datasets}
// -----------------------
namespace datasets {
    void collect(const std::string root, const std::string sub, std::vector<std::string>& paths, std::vector<std::string>& fnames);
    cv::Mat RGB_Loader(std::string& path);
    cv::Mat GRAY_Loader(std::string& path);
    cv::Mat LoadImageFromFile(const std::string& filename);

    // ----------------------------------------------------
    // namespace{datasets} -> class{Augmentation}
    // ----------------------------------------------------
    class Augmentation {
    private:
        int grid_size = 8;
        cv::Size resize;
        int stable_count = 0;
        cv::Mat stable_cache;
    public:
        std::tuple<cv::Mat, cv::Mat> generateAnomaly(std::string& file_path, cv::Size resize);
        cv::Mat generatePerlinNoise(cv::Mat& img);
        cv::Mat generatePerlinNoise2D(int width, int height, int res_x, int res_y);
        cv::Mat rand_augment(cv::Mat& img);
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
		Augmentation augmentor;
    public:
        SegmentImageWithPaths() {}
        SegmentImageWithPaths(const std::string root, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform, const std::string mode, cv::Size resize);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, int, cv::Mat, cv::Mat>& data);
        size_t size();
    };
}

#endif