#pragma once
#ifdef CLASSIFICATION_EXPORTS
#define CLASSIFICATION_DECLSPEC __declspec(dllexport)
#else
#define CLASSIFICATION_DECLSPEC __declspec(dllimport)
#endif

#include <iostream>                    // std::cout, std::cerr
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand, std::exit

// For External Library
#include <torch/torch.h>               // torch
#include <torch/script.h>
#include <opencv2/opencv.hpp>          // cv::Mat
#include <boost/program_options.hpp>   // boost::program_options

// For Original Header
#include "ini.h"
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // MC_ResNet
#include "./utils/transforms.hpp"              // transforms
#include "./utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "./utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths
#include "./utils/visualizer.hpp"              // visualizer
#include "./utils/progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
extern "C" CLASSIFICATION_DECLSPEC int mainClassification(int argc, const char* argv[], std::string file_path);

torch::Device Set_Device(mINI::INIStructure& ini);
void Set_Model_Params(mINI::INIStructure& ini, MC_ResNet& model, const std::string name);
std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num);
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], const std::string mode);
bool stringToBool(const std::string& str);

void train(mINI::INIStructure& ini, torch::Device& device, MC_ResNet& model, const std::vector<std::string> class_names);
void test(mINI::INIStructure& ini, torch::Device& device, MC_ResNet& model, const std::vector<std::string> class_names);
void valid(mINI::INIStructure& ini, DataLoader::ImageFolderClassesWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, MC_ResNet& model, const std::vector<std::string> class_names, const size_t epoch, visualizer::graph& writer, visualizer::graph& writer_accuracy, visualizer::graph& writer_each_accuracy);