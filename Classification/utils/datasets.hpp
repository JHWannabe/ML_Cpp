#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "../png++/png.hpp"
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// For Original Header
#include "transforms.hpp"

namespace fs = std::filesystem;


// -----------------------
// namespace{datasets}
// -----------------------
namespace datasets{

    // Function Prototype
    void collect(const std::string root, const std::string sub, std::vector<std::string> &paths, std::vector<std::string> &fnames);
    torch::Tensor Data1d_Loader(std::string &path);
    cv::Mat RGB_Loader(std::string &path);
    cv::Mat Index_Loader(std::string &path);
    std::tuple<torch::Tensor, torch::Tensor> BoundingBox_Loader(std::string &path);

    // ----------------------------------------------------------
    // namespace{datasets} -> class{ImageFolderClassesWithPaths}
    // ----------------------------------------------------------
    class ImageFolderClassesWithPaths{
    private:
        std::vector<transforms_Compose> transform;
        std::vector<std::string> paths, fnames;
        std::vector<size_t> class_ids;
    public:
        ImageFolderClassesWithPaths(){}
        ImageFolderClassesWithPaths(const std::string root, std::vector<transforms_Compose> &transform_, const std::vector<std::string> class_names);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string> &data);
        size_t size();
    };
}



#endif