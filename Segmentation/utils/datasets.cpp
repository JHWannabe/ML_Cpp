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
void datasets::collect(const std::string root, const std::string sub, std::vector<std::string> &paths, std::vector<std::string> &fnames){
    fs::path ROOT(root);
    for (auto &p : fs::directory_iterator(ROOT)){
        if (!fs::is_directory(p)){
            std::stringstream rpath, fname;
            rpath << p.path().string();
            fname << p.path().filename().string();
            paths.push_back(rpath.str());
            fnames.push_back(sub + fname.str());
        }
        else{
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
torch::Tensor datasets::Data1d_Loader(std::string &path){

    float data_one;
    std::ifstream ifs;
    std::vector<float> data_src;
    torch::Tensor data;

    // Get Data
    ifs.open(path);
    while (1){
        ifs >> data_one;
        if (ifs.eof()) break;
        data_src.push_back(data_one);
    }
    ifs.close();

    // Get Tensor
    data = torch::from_blob(data_src.data(), {(long int)data_src.size()}, torch::kFloat).clone();

    return data;

}

cv::Mat LoadImageFromFile(const std::string& filename) {
    // Read the file as a binary stream
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return cv::Mat();
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the file into a buffer
    std::vector<uchar> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    // Decode the image using OpenCV
    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}


// -----------------------------------------------
// namespace{datasets} -> function{RGB_Loader}
// -----------------------------------------------
cv::Mat datasets::RGB_Loader(std::string &path){
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
cv::Mat datasets::Index_Loader(std::string &path){
    size_t i, j;    
    size_t width, height;
    cv::Mat Index;
    png::image<png::index_pixel> Index_png(path);  // path ===> index image
    width = Index_png.get_width();
    height = Index_png.get_height();
    Index = cv::Mat(cv::Size(width, height), CV_32SC1);
    for (j = 0; j < height; j++){
        for (i = 0; i < width; i++){
            Index.at<int>(j, i) = (int)Index_png[j][i];
        }
    }
    return Index.clone();
}




// ----------------------------------------------------
// namespace{datasets} -> function{BoundingBox_Loader}
// ----------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> datasets::BoundingBox_Loader(std::string &path){
    
    FILE *fp;
    int state;
    long int id_data;
    float cx_data, cy_data, w_data, h_data;
    torch::Tensor id, cx, cy, w, h, coord;
    torch::Tensor ids, coords;
    std::tuple<torch::Tensor, torch::Tensor> BBs;

    if ((fp = fopen(path.c_str(), "r")) == NULL){
        std::cerr << "Error : Couldn't open the file '" << path << "'." << std::endl;
        std::exit(1);
    }

    state = 0;
    while (fscanf(fp, "%ld %f %f %f %f", &id_data, &cx_data, &cy_data, &w_data, &h_data) != EOF){

        id = torch::full({ 1 }, static_cast<int64_t>(id_data), torch::TensorOptions().dtype(torch::kLong));
        cx = torch::full({1, 1}, cx_data, torch::TensorOptions().dtype(torch::kFloat));  // cx{1,1}
        cy = torch::full({1, 1}, cy_data, torch::TensorOptions().dtype(torch::kFloat));  // cy{1,1}
        w = torch::full({1, 1}, w_data, torch::TensorOptions().dtype(torch::kFloat));  // w{1,1}
        h = torch::full({1, 1}, h_data, torch::TensorOptions().dtype(torch::kFloat));  // h{1,1}
        coord = torch::cat({cx, cy, w, h}, /*dim=*/1);  // cx{1,1} + cy{1,1} + w{1,1} + h{1,1} ===> coord{1,4}
        
        switch (state){
            case 0:
                ids = id;  // id{1} ===> ids{1}
                coords = coord;  // coord{1,4} ===> coords{1,4}
                state = 1;
                break;
            default:
                ids = torch::cat({ids, id}, /*dim=*/0);  // ids{i} + id{1} ===> ids{i+1}
                coords = torch::cat({coords, coord}, /*dim=*/0);  // coords{i,4} + coord{1,4} ===> coords{i+1,4}
        }

    }
    fclose(fp);

    if (ids.numel() > 0){
        ids = ids.contiguous().detach().clone();
        coords = coords.contiguous().detach().clone();
    }
    BBs = {ids, coords};  // {BB_n} (ids), {BB_n,4} (coordinates)

    return BBs;

}

// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderSegmentWithPaths} -> constructor
// -------------------------------------------------------------------------
datasets::ImageFolderSegmentWithPaths::ImageFolderSegmentWithPaths(const std::string root1, const std::string root2, std::vector<transforms_Compose> &transformI_, std::vector<transforms_Compose> &transformO_){

    datasets::collect(root1 + '/', "", this->paths1, this->fnames1);
    std::sort(this->paths1.begin(), this->paths1.end());
    std::sort(this->fnames1.begin(), this->fnames1.end());

    std::string f_png;
    std::string::size_type pos;
    for (auto &f : this->fnames1){
        if ((pos = f.find_last_of(".")) == std::string::npos){
            f_png = f + ".png";
        }
        else{
            f_png = f.substr(0, pos) + ".png";
        }
        std::string path2 = root2 + '/' + f_png;
        this->fnames2.push_back(f_png);
        this->paths2.push_back(path2);
    }

    this->transformI = transformI_;
    this->transformO = transformO_;

    try {
        png::image<png::rgb_pixel> rgbImage(paths2.at(0));
        //png::image<png::index_pixel> Index_png(paths2.at(0));
        png::palette pal = rgbImage.get_palette();
        for (auto& p : pal) {
            this->label_palette.push_back({ (unsigned char)p.red, (unsigned char)p.green, (unsigned char)p.blue });
        }
    }
    catch (const png::error& e) {
        std::cerr << "PNG error occurred: " << e.what() << std::endl;
    }


}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderSegmentWithPaths} -> function{get}
// -------------------------------------------------------------------------
void datasets::ImageFolderSegmentWithPaths::get(const size_t idx, std::tuple<torch::Tensor, torch::Tensor, std::string, std::string, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> &data){
    cv::Mat image_Mat1 = datasets::RGB_Loader(this->paths1.at(idx));
    //cv::Mat image_Mat2 = datasets::Index_Loader(this->paths2.at(idx));
    cv::Mat image_Mat2 = datasets::RGB_Loader(this->paths2.at(idx));
    torch::Tensor image1 = transforms::apply(this->transformI, image_Mat1);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    torch::Tensor image2 = transforms::apply(this->transformO, image_Mat2);  // Mat Image ==={Resize,ToTensor,etc.}===> Tensor Image
    std::string fname1 = this->fnames1.at(idx);
    std::string fname2 = this->fnames2.at(idx);
    data = {image1.detach().clone(), image2.detach().clone(), fname1, fname2, this->label_palette};
    return;
}


// -------------------------------------------------------------------------
// namespace{datasets} -> class{ImageFolderSegmentWithPaths} -> function{size}
// -------------------------------------------------------------------------
size_t datasets::ImageFolderSegmentWithPaths::size(){
    return this->fnames1.size();
}
