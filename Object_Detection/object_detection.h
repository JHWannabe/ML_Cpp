#pragma once
#ifdef OBJECTDETECTION_EXPORTS
#define OBJECTDETECTION_DECLSPEC __declspec(dllexport)
#else
#define OBJECTDETECTION_DECLSPEC __declspec(dllimport)
#endif

#include <iostream>                    // std::cout, std::cerr
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::istringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand, std::exit
#include <utility>                     // std::pair
#include <cmath>                       // std::pow

// For External Library
#include <torch/torch.h>               // torch
#include <torch/script.h>
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options

// For Original Header
#include "ini.h"
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // YOLOv3
#include "augmentation.hpp"            // YOLOAugmentation
#include "detector.hpp"                // YOLODetectors
#include "./utils/transforms.hpp"      // transforms
#include "./utils/dataloader.hpp"      // DataLoader::ImageFolderBBWithPaths
#include "./utils/visualizer.hpp"      // visualizer::graph
#include "./utils/datasets.hpp"        // datasets::ImageFolderBBWithPaths
#include "./utils/progress.hpp"        // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace F = torch::nn::functional;

// Function Prototype
extern "C" OBJECTDETECTION_DECLSPEC int mainObjectDetection(int argc, const char* argv[], std::string file_path);

torch::Device Set_Device(mINI::INIStructure& ini);
void Set_Model_Params(mINI::INIStructure& ini, YOLOv3& model, const std::string name);
std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num);
std::vector<std::vector<std::tuple<float, float>>> Set_Anchors(const std::string path, const size_t scales, const size_t na);
std::vector<std::tuple<long int, long int>> Set_Resizes(const std::string path, size_t& resize_step_max);
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], po::options_description& args, const std::string mode);
bool stringToBool(const std::string& str);
template <typename Optimizer, typename OptimizerOptions>
void Update_LR(Optimizer& optimizer, const float lr_init, const float lr_base, const float lr_decay1, const float lr_decay2, const size_t epoch, const float burnin_base, const float burnin_exp = 4.0);

void test(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transform, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors);
void train(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transformBB, std::vector<transforms_Compose>& transformI, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors, const std::vector<std::tuple<long int, long int>> resizes, const size_t resize_step_max);
void valid(mINI::INIStructure& ini, DataLoader::ImageFolderBBWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, YOLOv3& model, const std::vector<std::string> class_names, const size_t epoch, std::vector<visualizer::graph>& writer);
void detect(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformD, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors);
void demo(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformD, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors);