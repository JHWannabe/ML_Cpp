#pragma once
#ifdef SEGMENTATION_EXPORTS
#define SEGMENTATION_DECLSPEC __declspec(dllexport)
#else
#define SEGMENTATION_DECLSPEC __declspec(dllimport)
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

// For Original Header
#include "ini.h"
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // UNet
#include "./utils/transforms.hpp"              // transforms
#include "./utils/datasets.hpp"                // datasets::ImageFolderSegmentWithPaths
#include "./utils/dataloader.hpp"              // DataLoader::ImageFolderSegmentWithPaths
#include "./utils/visualizer.hpp"              // visualizer
#include "./utils/progress.hpp"                // progress



// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
extern "C" SEGMENTATION_DECLSPEC int mainSegmentation(int argc, const char* argv[], std::string file_path);

torch::Device Set_Device(mINI::INIStructure& ini);
void Set_Model_Params(mINI::INIStructure& ini, UNet& model, const std::string name);
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], po::options_description& args, const std::string mode);
bool stringToBool(const std::string& str);

void test(mINI::INIStructure& ini, torch::Device& device, UNet& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformO);
void train(mINI::INIStructure& ini, torch::Device& device, UNet& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformO);
void valid(mINI::INIStructure& ini, DataLoader::ImageFolderSegmentWithPaths& valid_dataloader, torch::Device& device, CEDiceLoss& criterion, UNet& model, const size_t epoch, visualizer::graph& writer);