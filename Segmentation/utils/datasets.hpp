#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <string>
#include <tuple>
#include <vector>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// For Original Header
#include "transforms.hpp"


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

    // ----------------------------------------------------
    // namespace{datasets} -> class{ImageFolderSegmentWithPaths}
    // ----------------------------------------------------
    class ImageFolderSegmentWithPaths{
    private:
        std::vector<transforms_Compose> transformI, transformO;
        std::vector<std::string> paths1, paths2, fnames1, fnames2;
        // std::vector<cv::Vec3b> label_palette;
        std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette;
    public:
        ImageFolderSegmentWithPaths(){}
        ImageFolderSegmentWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_);
        void get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data);
        size_t size();
    };
    
}



#endif