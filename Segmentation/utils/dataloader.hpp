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
namespace DataLoader{
    // -----------------------------------------------------
    // namespace{DataLoader} -> class{ImageFolderSegmentWithPaths}
    // -----------------------------------------------------
    class ImageFolderSegmentWithPaths{
    private:
        datasets::ImageFolderSegmentWithPaths dataset;
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
    public:
        ImageFolderSegmentWithPaths(){}
        ImageFolderSegmentWithPaths(datasets::ImageFolderSegmentWithPaths &dataset_, const size_t batch_size_=1, const bool shuffle_=false, const size_t num_workers_=0, const bool pin_memory_=false, const bool drop_last_=false);
        bool operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data);
        void reset();
        size_t get_count_max();
    };
}

#endif