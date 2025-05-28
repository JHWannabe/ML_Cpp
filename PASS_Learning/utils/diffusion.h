#pragma once
#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

cv::Mat stableDiffusion(std::string& file_path);
void initializePython();