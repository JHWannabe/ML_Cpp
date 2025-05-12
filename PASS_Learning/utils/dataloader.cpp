#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cmath>
// For External Library
#include <torch/torch.h>
#include <omp.h>
// For Original Header
#include "datasets.hpp"
#include "dataloader.hpp"


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{SegmentImageWithPaths} -> constructor
// --------------------------------------------------------------------
DataLoader::SegmentImageWithPaths::SegmentImageWithPaths(datasets::SegmentImageWithPaths& dataset_, const size_t batch_size_, const bool shuffle_, const size_t num_workers_, const bool pin_memory_, const bool drop_last_, const std::string mode) {

    this->dataset = dataset_;
    this->batch_size = batch_size_;
    this->shuffle = shuffle_;
    this->num_workers = num_workers_;
    this->pin_memory = pin_memory_;
    this->drop_last = drop_last_;
    this->mode = mode;

    this->size = this->dataset.size();
    this->idx = std::vector<size_t>(this->size);
    for (size_t i = 0; i < this->size; i++) {
        this->idx.at(i) = i;
    }

    this->count = 0;
    if (this->drop_last) {
        this->count_max = std::floor((float)this->size / (float)this->batch_size);
        if ((this->count_max == 0) && (this->size > 0)) {
            this->count_max = 1;
        }
    }
    else {
        this->count_max = std::ceil((float)this->size / (float)this->batch_size);
    }

    this->mt.seed(std::rand());

}


// --------------------------------------------------------------------
// namespace{DataLoader} -> class{SegmentImageWithPaths} -> operator
// --------------------------------------------------------------------
bool DataLoader::SegmentImageWithPaths::operator()(std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<int>, std::vector<cv::Mat>, std::vector<cv::Mat>>& data) {

    // (0) Initialization and Declaration
    size_t i;
    size_t idx_start = this->batch_size * this->count;
    size_t idx_end = std::min(this->size, (idx_start + this->batch_size));
    size_t mini_batch_size = idx_end - idx_start;
    torch::Tensor data1, data2, tensor1, tensor2;
    std::vector<std::string> data3;
    std::vector<int> data4;
    std::vector<cv::Mat> data5, data6;
    std::tuple<torch::Tensor, torch::Tensor, std::string, int, cv::Mat, cv::Mat> group;
    std::tuple<torch::Tensor, torch::Tensor, std::string, int, cv::Mat, cv::Mat>* data_before;

    // (1) Special Handling on Certain Count
    if ((this->count == 0) && this->shuffle) {
        std::shuffle(this->idx.begin(), this->idx.end(), this->mt);
    }
    else if (this->count == this->count_max) {
        this->count = 0;
        return false;
    }

    // (2) Get Mini Batch Data
    data_before = new std::tuple<torch::Tensor, torch::Tensor, std::string, int, cv::Mat, cv::Mat>[mini_batch_size];
    // (2.1) Get Mini Batch Data using Single Thread
    if (this->num_workers == 0) {
        for (i = 0; i < mini_batch_size; i++) {
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }
    // (2.2) Get Mini Batch Data using Multi Thread
    else {
        omp_set_num_threads(this->num_workers);
        #pragma omp parallel for
        for (i = 0; i < mini_batch_size; i++) {
            this->dataset.get(this->idx.at(idx_start + i), data_before[i]);
        }
    }

    // (3) Organize Data
    data1 = std::get<0>(data_before[0]);
    data1 = torch::unsqueeze(data1, /*dim=*/0);
    data2 = std::get<1>(data_before[0]);
    data2 = torch::unsqueeze(data2, /*dim=*/0);
    data3.push_back(std::get<2>(data_before[0]));
    data4.push_back(std::get<3>(data_before[0]));
    data5.push_back(std::get<4>(data_before[0]));
    data6.push_back(std::get<5>(data_before[0]));
    for (i = 1; i < mini_batch_size; i++) {
        group = data_before[i];
        tensor1 = std::get<0>(group);
        tensor1 = torch::unsqueeze(tensor1, /*dim=*/0);  // {C,H,W} ===> {1,C,H,W}
        data1 = torch::cat({ data1, tensor1 }, /*dim=*/0);  // {i,C,H,W} + {1,C,H,W} ===> {i+1,C,H,W}
        tensor2 = std::get<1>(group);
        tensor2 = torch::unsqueeze(tensor2, /*dim=*/0);  // {H,W} ===> {1,H,W}
        data2 = torch::cat({ data2, tensor2 }, /*dim=*/0);  // {i,H,W} + {1,H,W} ===> {i+1,H,W}
        data3.push_back(std::get<2>(group));
        data4.push_back(std::get<3>(group));
        data5.push_back(std::get<4>(group));
        data6.push_back(std::get<5>(group));
    }
    data1 = data1.contiguous().detach().clone();
    data2 = data2.contiguous().detach().clone();

    // (4) Pin
    if (this->pin_memory) {
        data1 = data1.pin_memory();
        data2 = data2.pin_memory();
    }

    // Post Processing
    this->count++;

	data = { data1, data2, data3, data4, data5, data6 };  // {N,C,H,W} (image), {N,H,W} (label), {N} (fnames), {N} (y_true), {H,W} (image), {H,W} (label)
    delete[] data_before;

    // End Processings
    return true;

}


// -------------------------------------------------------------------------------
// namespace{DataLoader} -> class{SegmentImageWithPaths} -> function{reset}
// -------------------------------------------------------------------------------
void DataLoader::SegmentImageWithPaths::reset() {
    this->count = 0;
    return;
}


// ---------------------------------------------------------------------------------------
// namespace{DataLoader} -> class{SegmentImageWithPaths} -> function{get_count_max}
// ---------------------------------------------------------------------------------------
size_t DataLoader::SegmentImageWithPaths::get_count_max() {
    return this->count_max;
}
