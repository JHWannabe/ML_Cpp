#pragma once
#ifdef PASSLEARNING_EXPORTS
#define PASSLEARNING_DECLSPEC __declspec(dllexport)
#else
#define PASSLEARNING_DECLSPEC __declspec(dllimport)
#endif

#include <iostream>                    // std::cout
#include <fstream>                     // std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <chrono>                      // std::chrono
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <utility>                     // std::pair

// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options
#include <c10/util/Exception.h>
#include <pybind11/embed.h>    // gil_scoped_release

// For Original Header
#include "ini.h"
#include "model/networks.h"                // Model
#include "utils/losses.hpp"                    // Loss
#include "utils/transforms.hpp"              // transforms
#include "utils/datasets.hpp"                // datasets::ImageFolderSegmentWithPaths
#include "utils/dataloader.hpp"              // DataLoader::ImageFolderSegmentWithPaths
#include "utils/visualizer.hpp"              // visualizer
#include "utils/progress.hpp"                // progress
#include "utils/scheduler.hpp"

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace py = pybind11;

// Function Prototype
extern "C" PASSLEARNING_DECLSPEC int mainPASSLearning(int argc, const char* argv[], std::string file_path);

torch::Device Set_Device(mINI::INIStructure& ini);
void Set_Model_Params(mINI::INIStructure& ini, std::shared_ptr<Supervised>& model, const std::string name);
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], po::options_description& args, const std::string mode);
bool stringToBool(const std::string& str);

void test(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform);
void train(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform);
void valid(mINI::INIStructure& ini, DataLoader::SegmentImageWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, std::shared_ptr<Supervised>& model, const size_t epoch, visualizer::graph& writer);

struct Metrics {
	double recall;
	double precision;
	double f1_score;
	int FN;
};

int count_images_in_label_folder(const std::string& base_path);
Metrics compute_metrics_from_confusion_matrix(const std::vector<std::vector<int>>& cm);
std::vector<std::vector<int>> compute_confusion_matrix_from_dirs(
	const std::string& goodPath,
	const std::string& ngPath,
	const std::string& overkillPath,
	const std::string& notfoundPath);