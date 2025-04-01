#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
// #include <png.h>
#include "../png++/png.hpp"
// For Original Header
#include "transforms.hpp"
#include "datasets.hpp"

namespace fs = std::filesystem;

// -----------------------------------------------
// namespace{datasets} -> function{collect}
// -----------------------------------------------
void datasets::collect(const std::string root, const std::string sub, std::vector<std::string>& paths, std::vector<std::string>& fnames) {
    fs::path ROOT(root);
    for (auto& p : fs::directory_iterator(ROOT)) {
        if (!fs::is_directory(p)) {
            std::stringstream rpath, fname;
            rpath << p.path().string();
            fname << p.path().filename().string();
            paths.push_back(rpath.str());
            fnames.push_back(sub + fname.str());
        }
        else {
            std::stringstream subsub;
            subsub << p.path().filename().string();
            datasets::collect(root + '/' + subsub.str(), sub + subsub.str() + '/', paths, fnames);
        }
    }
    return;
}


// -----------------------------------------------
// namespace{datasets} -> function{Data1d_Loader}
// -----------------------------------------------
torch::Tensor datasets::Data1d_Loader(std::string& path) {

    float data_one;
    std::ifstream ifs;
    std::vector<float> data_src;
    torch::Tensor data;

    // Get Data
    ifs.open(path);
    while (1) {
        ifs >> data_one;
        if (ifs.eof()) break;
        data_src.push_back(data_one);
    }
    ifs.close();

    // Get Tensor
    data = torch::from_blob(data_src.data(), { (long int)data_src.size() }, torch::kFloat).clone();

    return data;

}


cv::Mat LoadImageFromFile(const std::string& filename) {
    // 파일을 바이너리로 읽기
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return cv::Mat();
    }

    // 파일 크기 구하기
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 바이트 배열로 읽기
    std::vector<uchar> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    // OpenCV를 이용하여 디코딩
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

// -----------------------------------------------
// namespace{datasets} -> function{RGB_Loader}
// -----------------------------------------------
cv::Mat datasets::RGB_Loader(std::string& path) {
    cv::Mat BGR, RGB;
    BGR = LoadImageFromFile(path);

    if (BGR.empty()) {
        std::cerr << "Error : Couldn't open the image '" << path << "'." << std::endl;
        std::exit(1);
    }
    cv::cvtColor(BGR, RGB, cv::COLOR_BGR2RGB);  // {0,1,2} = {B,G,R} ===> {0,1,2} = {R,G,B}
    return RGB.clone();
}


// -----------------------------------------------
// namespace{datasets} -> function{Index_Loader}
// -----------------------------------------------
cv::Mat datasets::Index_Loader(std::string& path) {
    size_t i, j;
    size_t width, height;
    cv::Mat Index;
    png::image<png::index_pixel> Index_png(path);  // path ===> index image
    width = Index_png.get_width();
    height = Index_png.get_height();
    Index = cv::Mat(cv::Size(width, height), CV_32SC1);
    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i++) {
            Index.at<int>(j, i) = (int)Index_png[j][i];
        }
    }
    return Index.clone();
}


// ----------------------------------------------------
// namespace{datasets} -> function{BoundingBox_Loader}
// ----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> datasets::BoundingBox_Loader(std::string& path) {

    FILE* fp;
    int state;
    long int id_data;
    float cx_data, cy_data, w_data, h_data;
    torch::Tensor id, cx, cy, w, h, coord;
    torch::Tensor ids, coords;
    std::tuple<torch::Tensor, torch::Tensor> BBs;

    if ((fp = fopen(path.c_str(), "r")) == NULL) {
        std::cerr << "Error : Couldn't open the file '" << path << "'." << std::endl;
        std::exit(1);
    }

    state = 0;
    while (fscanf(fp, "%ld %f %f %f %f", &id_data, &cx_data, &cy_data, &w_data, &h_data) != EOF) {

        id = torch::full({ 1 }, id_data, torch::TensorOptions().dtype(torch::kLong));  // id{1}
        cx = torch::full({ 1, 1 }, cx_data, torch::TensorOptions().dtype(torch::kFloat));  // cx{1,1}
        cy = torch::full({ 1, 1 }, cy_data, torch::TensorOptions().dtype(torch::kFloat));  // cy{1,1}
        w = torch::full({ 1, 1 }, w_data, torch::TensorOptions().dtype(torch::kFloat));  // w{1,1}
        h = torch::full({ 1, 1 }, h_data, torch::TensorOptions().dtype(torch::kFloat));  // h{1,1}
        coord = torch::cat({ cx, cy, w, h }, /*dim=*/1);  // cx{1,1} + cy{1,1} + w{1,1} + h{1,1} ===> coord{1,4}

        switch (state) {
        case 0:
            ids = id;  // id{1} ===> ids{1}
            coords = coord;  // coord{1,4} ===> coords{1,4}
            state = 1;
            break;
        default:
            ids = torch::cat({ ids, id }, /*dim=*/0);  // ids{i} + id{1} ===> ids{i+1}
            coords = torch::cat({ coords, coord }, /*dim=*/0);  // coords{i,4} + coord{1,4} ===> coords{i+1,4}
        }

    }
    fclose(fp);

    if (ids.numel() > 0) {
        ids = ids.contiguous().detach().clone();
        coords = coords.contiguous().detach().clone();
    }
    BBs = { ids, coords };  // {BB_n} (ids), {BB_n,4} (coordinates)

    return BBs;

}



// -------------------------------------------------------------------------
// namespace{datasets} -> class{SegmentImageWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::SegmentImageWithPaths::SegmentImageWithPaths(const std::string root, std::vector<transforms_Compose>& transform, const std::string mode) {
    datasets::collect(root, "", this->paths, this->fnames);
    std::sort(this->paths.begin(), this->paths.end());
    std::sort(this->fnames.begin(), this->fnames.end());
    this->transform = transform;
    this->mode = mode;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{SegmentImageWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::SegmentImageWithPaths::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, int>& data) {
    if (this->mode == "test" || this->mode == "valid") {
        cv::Mat image_Mat = datasets::RGB_Loader(this->paths.at(idx));
        std::string fname = this->fnames.at(idx);
        if (this->paths.at(idx).find("NG") != std::string::npos) {
            this->y_true = 0;
        }

        std::string label_path = this->paths.at(idx);
        size_t pos;
        if ((pos = label_path.find("bmp")) != std::string::npos) {
            label_path.replace(pos, 4, "jpg");
        }
        while ((pos = label_path.find("origin")) != std::string::npos) {
            label_path.replace(pos, 6, "label");
        }
        cv::Mat Label_Mat = datasets::RGB_Loader(label_path);
        cv::Mat Gray_Label_Mat;
        cv::cvtColor(Label_Mat, Gray_Label_Mat, cv::COLOR_RGB2GRAY);

        torch::Tensor image_tensor = transforms::apply(this->transform, image_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
        torch::Tensor label_tensor = transforms::apply(this->transform, Gray_Label_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image

        data = { image_tensor.detach().clone(), label_tensor.detach().clone(), fname, y_true };
    }
    else if (this->mode == "super") {
        cv::Mat image_Mat = datasets::RGB_Loader(this->paths.at(idx));
        std::string fname = this->fnames.at(idx);

        std::string label_path = this->paths.at(idx);
        size_t pos;
        if ((pos = label_path.find("bmp")) != std::string::npos) {
            label_path.replace(pos, 4, "jpg");
        }
        while ((pos = label_path.find("origin")) != std::string::npos) {
            label_path.replace(pos, 6, "label");
        }
        cv::Mat Label_Mat = datasets::RGB_Loader(label_path);
        cv::Mat Gray_Label_Mat;
        cv::cvtColor(Label_Mat, Gray_Label_Mat, cv::COLOR_RGB2GRAY);

        torch::Tensor image_tensor = transforms::apply(this->transform, image_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
        torch::Tensor label_tensor = transforms::apply(this->transform, Gray_Label_Mat);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image

        data = { image_tensor.detach().clone(), label_tensor.detach().clone(), fname, y_true };
    }
    else if (this->mode == "unsuper") {
        cv::Mat image_Mat = datasets::RGB_Loader(this->paths.at(idx));
        cv::Mat label_Mat = cv::Mat::zeros(image_Mat.size(), CV_8UC1);  
        std::string fname = this->fnames.at(idx);
        std::vector<std::pair<cv::Mat, cv::Mat>> result;

        AnomalyGenerator anomalyGenerator = AnomalyGenerator();
        result = anomalyGenerator.generateAnomaly(image_Mat);

        torch::Tensor image_tensor = transforms::apply(this->transform, result[0].first);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
        torch::Tensor label_tensor = transforms::apply(this->transform, result[0].second);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image

        data = { image_tensor.detach().clone(), label_tensor.detach().clone(), fname, y_true };
    }
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{SegmentImageWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::SegmentImageWithPaths::size() {
    return this->fnames.size();
}





cv::Mat AnomalyGenerator::generateTargetForegroundMask(cv::Mat img)
{
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_RGB2GRAY);

    cv::Mat targetBackgroundMask;
    cv::threshold(imgGray, targetBackgroundMask, 0, 255, cv::THRESH_BINARY_INV);

    targetBackgroundMask.convertTo(targetBackgroundMask, CV_32S);

    cv::Mat targetForegroundMask = -(targetBackgroundMask - 1);

    return targetForegroundMask;
}

cv::Mat AnomalyGenerator::anomalySource(cv::Mat img)
{
    std::vector<std::string> texture_source_file_list;
    if (texture_source_file_list.empty()) {
        throw std::runtime_error("Error: texture_source_file_list is empty!");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, texture_source_file_list.size() - 1);
    size_t idx = dis(gen);

    cv::Mat texture_source_img = cv::imread(texture_source_file_list[idx]);
    if (texture_source_img.empty()) {
        throw std::runtime_error("Error: Failed to load texture image: " + texture_source_file_list[idx]);
    }

    cv::cvtColor(texture_source_img, texture_source_img, cv::COLOR_BGR2RGB);

    cv::resize(texture_source_img, texture_source_img, resize);
    texture_source_img.convertTo(texture_source_img, CV_32F);

    return texture_source_img;
}

std::vector<std::pair<cv::Mat, cv::Mat>> AnomalyGenerator::generateAnomaly(cv::Mat img)
{
    // Step 1: Generate masks
    cv::Mat TargetForegroundMask = this->generateTargetForegroundMask(img);
    cv::Mat PerlinNoiseMask = this->generatePerlinNoiseMask(img);
    TargetForegroundMask.convertTo(TargetForegroundMask, CV_8U);
    PerlinNoiseMask.convertTo(PerlinNoiseMask, CV_8U);

    cv::Mat mask;
    cv::bitwise_and(PerlinNoiseMask, TargetForegroundMask, mask);

    cv::imwrite("mask_output.png", mask * 255);

    // Expand mask to 3 channels and normalize
    cv::Mat mask_expanded;
    cv::cvtColor(mask, mask_expanded, cv::COLOR_GRAY2BGR);
    mask_expanded.convertTo(mask_expanded, CV_32F, 1.0 / 255.0);

    // Step 2: Generate anomaly source
    cv::Mat anomalySourceImg = this->anomalySource(img);
    anomalySourceImg.convertTo(anomalySourceImg, CV_32F);
    img.convertTo(img, CV_32F);

    // Step 3: Blend the image and anomaly source
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.15, 1.0);
    float factor = dis(gen);

    anomalySourceImg = factor * (mask_expanded.mul(anomalySourceImg)) + (1 - factor) * (mask_expanded.mul(img));

    // Final blending
    anomalySourceImg = (1 - mask_expanded).mul(img) + anomalySourceImg;

    // Convert back to CV_8U (if needed)
    anomalySourceImg.convertTo(anomalySourceImg, CV_8U);
    mask.convertTo(mask, CV_8U);

    // Return the result
    return { { anomalySourceImg, mask } };
}


cv::Mat AnomalyGenerator::generatePerlinNoiseMask(cv::Mat img) {
    int minScale = 1;
    int maxScale = 6;
    float threshold = 0.5;
    int width = img.cols, height = img.rows;

    int scaleX = 1 << (rand() % (maxScale - minScale + 1) + minScale);
    int scaleY = 1 << (rand() % (maxScale - minScale + 1) + minScale);

    cv::Mat perlinNoise = this->rand_perlin_2d_np(cv::Size(width, height), cv::Size(scaleX, scaleY));

    cv::Point2f center(width / 2.0F, height / 2.0F);
    double angle = (rand() % 180) - 90; 
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(perlinNoise, perlinNoise, rotationMatrix, perlinNoise.size());

    cv::Mat mask;
    cv::threshold(perlinNoise, mask, threshold, 1, cv::THRESH_BINARY);

    return mask;
}

cv::Mat AnomalyGenerator::rand_perlin_2d_np(cv::Size shape, cv::Size res) {

    cv::Mat grid(shape, CV_32FC2);
    cv::Mat angles(res.height + 1, res.width + 1, CV_32F);
    cv::Mat gradients(res.height + 1, res.width + 1, CV_32FC2);

    cv::Point2f delta(res.width / static_cast<float>(shape.width), res.height / static_cast<float>(shape.height));
    cv::Point2i d(shape.width / res.width, shape.height / res.height);

    for (int i = 0; i < shape.height; ++i) {
        for (int j = 0; j < shape.width; ++j) {
            grid.at<cv::Vec2f>(i, j) = cv::Vec2f((j * delta.x) - std::floor(j * delta.x), (i * delta.y) - std::floor(i * delta.y));
        }
    }

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0, 2 * CV_PI);
    for (int i = 0; i < res.height + 1; ++i) {
        for (int j = 0; j < res.width + 1; ++j) {
            angles.at<float>(i, j) = dist(gen);
            gradients.at<cv::Vec2f>(i, j) = cv::Vec2f(std::cos(angles.at<float>(i, j)), std::sin(angles.at<float>(i, j)));
        }
    }

    auto tile_grads = [&](cv::Range slice1, cv::Range slice2) {
        cv::Mat result;
        cv::repeat(gradients(slice1, slice2), d.y, d.x, result);
        return result;
        };

    auto dot = [&](const cv::Mat& grad, const cv::Vec2f& shift) {
        cv::Mat result(shape, CV_32F);
        for (int i = 0; i < shape.height; ++i) {
            for (int j = 0; j < shape.width; ++j) {
                cv::Vec2f g = grad.at<cv::Vec2f>(i, j);
                cv::Vec2f p = grid.at<cv::Vec2f>(i, j) + shift;
                result.at<float>(i, j) = g.dot(p);
            }
        }
        return result;
        };

    cv::Mat n00 = dot(tile_grads(cv::Range(0, res.height), cv::Range(0, res.width)), cv::Vec2f(0, 0));
    cv::Mat n10 = dot(tile_grads(cv::Range(1, res.height + 1), cv::Range(0, res.width)), cv::Vec2f(-1, 0));
    cv::Mat n01 = dot(tile_grads(cv::Range(0, res.height), cv::Range(1, res.width + 1)), cv::Vec2f(0, -1));
    cv::Mat n11 = dot(tile_grads(cv::Range(1, res.height + 1), cv::Range(1, res.width + 1)), cv::Vec2f(-1, -1));

    cv::Mat t(shape, CV_32FC2);
    for (int i = 0; i < shape.height; ++i) {
        for (int j = 0; j < shape.width; ++j) {
            t.at<cv::Vec2f>(i, j) = cv::Vec2f(this->fade(grid.at<cv::Vec2f>(i, j)[0]), this->fade(grid.at<cv::Vec2f>(i, j)[1]));
        }
    }

    cv::Mat nx0(shape, CV_32F), nx1(shape, CV_32F), nxy(shape, CV_32F);
    for (int i = 0; i < shape.height; ++i) {
        for (int j = 0; j < shape.width; ++j) {
            nx0.at<float>(i, j) = this->lerp(n00.at<float>(i, j), n10.at<float>(i, j), t.at<cv::Vec2f>(i, j)[0]);
            nx1.at<float>(i, j) = this->lerp(n01.at<float>(i, j), n11.at<float>(i, j), t.at<cv::Vec2f>(i, j)[0]);
            nxy.at<float>(i, j) = this->lerp(nx0.at<float>(i, j), nx1.at<float>(i, j), t.at<cv::Vec2f>(i, j)[1]);
        }
    }

    return nxy * std::sqrt(2.0f);
}

