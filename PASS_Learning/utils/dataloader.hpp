#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <tuple>
#include <vector>
#include <random>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "datasets.hpp"

// -----------------------
// namespace{DataLoader}
// -----------------------
namespace DataLoader {
    // -----------------------------------------------------
    // namespace{DataLoader} -> class{SegmentImageWithPaths}
    // -----------------------------------------------------
    class SegmentImageWithPaths {
    private:
        datasets::SegmentImageWithPaths dataset;
        size_t batch_size;
        bool shuffle;
        size_t num_workers;
        bool pin_memory;
        bool drop_last;
        size_t size;
        std::vector<size_t> idx;
        size_t count;
        size_t count_max;
        std::mt19937 mt;
        std::string mode;
    public:
        SegmentImageWithPaths() {}
        SegmentImageWithPaths(datasets::SegmentImageWithPaths& dataset_, const size_t batch_size_ = 1, const bool shuffle_ = false, const size_t num_workers_ = 0, const bool pin_memory_ = false, const bool drop_last_ = false, const std::string mode = "");
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<int>, std::vector<cv::Mat>, std::vector<cv::Mat>>& data);
        void reset();
        size_t get_count_max();
    };

}

#endif