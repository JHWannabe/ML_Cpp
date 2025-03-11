#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
#include <torch/script.h>
#include <boost/program_options.hpp>   // boost::program_options
#include "ini.h"

// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // MC_ResNet
#include "./utils/transforms.hpp"              // transforms_Compose
#include "./utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "./utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths
#include "./utils/visualizer.hpp"              // visualizer
#include "./utils/progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;

// Function Prototype
void valid(mINI::INIStructure& ini, DataLoader::ImageFolderClassesWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, MC_ResNet& model, const std::vector<std::string> class_names, const size_t epoch, visualizer::graph& writer, visualizer::graph& writer_accuracy, visualizer::graph& writer_each_accuracy);
bool stringToBool(const std::string& str);

class CosineAnnealingLR {
public:
    CosineAnnealingLR(torch::optim::Optimizer& optimizer, double T_max, double eta_min, double eta_max)
        : optimizer_(optimizer), T_max_(T_max), eta_min_(eta_min), eta_max_(eta_max), last_epoch_(-1) {
    }

    void step(int epoch) {
        // Cosine annealing 공식에 따른 학습률 계산
        double lr = eta_min_ + 0.5 * (eta_max_ - eta_min_) * (1 + std::cos(M_PI * epoch / T_max_));

        // 현재 학습률을 옵티마이저에 반영
        for (auto& group : optimizer_.param_groups()) {
            group.options().set_lr(lr);
        }

        // 마지막 에폭 갱신
        last_epoch_ = epoch;
    }

private:
    torch::optim::Optimizer& optimizer_;
    double T_max_;
    double eta_min_;
    double eta_max_;
    int last_epoch_;
};

// -------------------
// Training Function
// -------------------
//void train(po::variables_map &vm, torch::Device &device, MC_ResNet &model, std::vector<transforms_Compose> &transform, const std::vector<std::string> class_names){
void train(mINI::INIStructure& ini, torch::Device& device, MC_ResNet& model, std::vector<transforms_Compose>& transform, const std::vector<std::string> class_names) {
    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, path;
    std::string dataroot, valid_dataroot;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss, valid_loss, valid_accuracy, valid_each_accuracy;
    progress::display* show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataroot = "./datasets/" + ini["General"]["dataset"] + "/" + ini["Training"]["train_dir"];
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, std::stol(ini["Training"]["batch_size"]), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if (stringToBool(ini["Validation"]["valid"])) {
        valid_dataroot = "./datasets/" + ini["General"]["dataset"] + "/" + ini["Validation"]["valid_dir"];
        valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
        valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, std::stol(ini["Validation"]["valid_batch_size"]), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(std::stof(ini["Network"]["lr"])).weight_decay(1e-5).betas({ std::stof(ini["Network"]["beta1"]), std::stof(ini["Network"]["beta2"]) }));

    //// CosineAnnealingLR 설정
    //double T_max = 100; // 한 주기 내에서 학습률이 변화하는 기간(에폭 수)
    //double eta_min = 1e-6; // 최소 학습률
    //double eta_max = 1e-3; // 최대 학습률
    //CosineAnnealingLR scheduler(optimizer, T_max, eta_min, eta_max);


    // (4) Set Loss Function
    auto criterion = Loss();

    // (5) Make Directories
    checkpoint_dir = "checkpoints/" + ini["General"]["dataset"];
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{ "Classification" });
    if (stringToBool(ini["Validation"]["valid"])) {
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{ "Classification" });
        valid_accuracy = visualizer::graph(path, /*gname_=*/"valid_accuracy", /*label_=*/{ "Accuracy" });
        valid_each_accuracy = visualizer::graph(path, /*gname_=*/"valid_each_accuracy", /*label_=*/class_names);
    }

    // (7) Get Weights and File Processing
    if (ini["Training"]["train_load_epoch"] == "") {
        if (stringToBool(ini["Training"]["pre_trained"])) {
            torch::jit::script::Module traced_model;
            std::string pre_trained_dir = checkpoint_dir + ini["Training"]["pre_trained_dir"];
            try {
                traced_model = torch::jit::load(pre_trained_dir);
            }
            catch (const c10::Error& e) {
                std::cerr << "Error loading the traced model. \n" + e.msg();
            }

            //// traced_model의 파라미터 출력
            //for (const auto& pair : traced_model.named_parameters()) {
            //    std::cout << pair.name << std::endl;
            //}

            //// model의 파라미터 출력
            //for (const auto& pair : model->named_parameters()) {
            //    std::cout << pair.key() << std::endl;
            //}


            for (const auto& pair : traced_model.named_parameters()) {
                auto param_name = pair.name;
                auto param_tensor = pair.value;

                // model의 named_parameters()을 통해 param_name에 해당하는 파라미터를 찾음
                auto params = model->named_parameters();
                // std::find_if를 사용하여 param_name이 일치하는 항목 찾기
                auto it = std::find_if(params.begin(), params.end(),
                    [&](const auto& p) { return p.key() == param_name; });

                if (it != params.end()) {
                    // 파라미터 크기 비교
                    if (it->value().sizes() != param_tensor.sizes()) {
                        std::cerr << "Shape mismatch for " << param_name
                            << ": model size: " << it->value().sizes()
                            << ", traced size: " << param_tensor.sizes() << std::endl;
                        continue;
                    }

                    // 파라미터 복사
                    try {
                        torch::NoGradGuard no_grad;  // 그래디언트 추적 비활성화
                        it->value().copy_(param_tensor.detach());  // in-place 연산 방지
                    }
                    catch (const c10::Error& e) {
                        std::cerr << "Error parameter copy for " << param_name << ": " + e.msg() << std::endl;
                    }
                }
                else {
                    std::cerr << "Parameter " << param_name << " not found in model." << std::endl;
                }
            }
        }
        else {
            model->init();
            ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
            if (stringToBool(ini["Validation"]["valid"])) {
                init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
                init.close();
            }
        }
        start_epoch = 0;
    }
    else {
        path = checkpoint_dir + "/models/epoch_" + ini["Training"]["train_load_epoch"] + ".pth";  torch::load(model, path, device);
        path = checkpoint_dir + "/optims/epoch_" + ini["Training"]["train_load_epoch"] + ".pth";  torch::load(optimizer, path, device);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::app);
        ofs << std::endl << std::endl;
        if (ini["Training"]["train_load_epoch"] == "latest") {
            infoi.open(checkpoint_dir + "/models/info.txt", std::ios::in);
            std::getline(infoi, buff);
            infoi.close();
            latest = "";
            for (auto& c : buff) {
                if (('0' <= c) && (c <= '9')) {
                    latest += c;
                }
            }
            start_epoch = std::stoi(latest);
        }
        else {
            start_epoch = std::stoi(ini["Training"]["train_load_epoch"]);
        }
    }

    // (8) Display Date
    date = progress::current_date();
    date = progress::separator_center("Train Loss (" + date + ")");
    std::cout << std::endl << std::endl << date << std::endl;
    ofs << date << std::endl;


    // -----------------------------------
    // a2. Training Model
    // -----------------------------------

    // (1) Set Parameters
    start_epoch++;
    total_iter = dataloader.get_count_max();
    total_epoch = std::stol(ini["Training"]["epochs"]);

    //float initial_lr = std::stof(ini["Network"]["lr"]);  // 초기 학습률
    //int step_size = 10;       // 학습률 감소 주기
    //float gamma = 0.1;        // 학습률 감소 비율

    // (2) Training per Epoch
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++) {

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{ epoch, total_epoch }, /*loss_=*/{ "classify" });


        // -----------------------------------
        // 학습률 스케줄러: 에폭마다 학습률 업데이트
        // -----------------------------------
        //if (epoch % step_size == 0 && epoch > 0) {
        //    float new_lr = initial_lr * std::pow(gamma, epoch / step_size);
        //    for (auto& group : optimizer.param_groups()) {
        //        group.options().set_lr(new_lr);
        //    }
        //}

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)) {

            // -----------------------------------
            // c1. ResNet Training Phase
            // -----------------------------------
            image = std::get<0>(mini_batch).to(device);
            label = std::get<1>(mini_batch).to(device);
            output = model->forward(image);
            loss = criterion(output, label);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            show_progress->increment(/*loss_value=*/{ loss.item().toFloat() });
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "classify:" << loss.item<float>() << "(ave:" << show_progress->get_ave(0) << ')' << std::endl;
        }

        // CosineAnnealingLR 스케줄러 업데이트
        // scheduler.step(epoch);
        std::cout << "Epoch [" << epoch << "] Learning Rate: "
            << optimizer.param_groups()[0].options().get_lr() << std::endl;

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train_loss.plot(/*base=*/epoch, /*value=*/show_progress->get_ave());
        delete show_progress;

        // -----------------------------------
        // b3. Validation Mode
        // -----------------------------------
        if (stringToBool(ini["Validation"]["valid"]) && ((epoch - 1) % std::stol(ini["Validation"]["valid_freq"]) == 0)) {
            valid(ini, valid_dataloader, device, criterion, model, class_names, epoch, valid_loss, valid_accuracy, valid_each_accuracy);
        }

        // -----------------------------------
        // b4. Save Model Weights and Optimizer Parameters
        // -----------------------------------
        if (epoch % std::stol(ini["Training"]["save_epoch"]) == 0) {
            path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + ".pth";  torch::save(model, path);
            path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + ".pth";  torch::save(optimizer, path);
        }
        path = checkpoint_dir + "/models/epoch_latest.pth";  torch::save(model, path);
        path = checkpoint_dir + "/optims/epoch_latest.pth";  torch::save(optimizer, path);
        infoo.open(checkpoint_dir + "/models/info.txt", std::ios::out);
        infoo << "latest = " << epoch << std::endl;
        infoo.close();

        // -----------------------------------
        // b5. Show Elapsed Time
        // -----------------------------------
        if (epoch % 10 == 0) {

            // -----------------------------------
            // c1. Get Output String
            // -----------------------------------
            ss.str(""); ss.clear(std::stringstream::goodbit);
            irreg_progress.nab(epoch);
            ss << "elapsed = " << irreg_progress.get_elap() << '(' << irreg_progress.get_sec_per() << "sec/epoch)   ";
            ss << "remaining = " << irreg_progress.get_rem() << "   ";
            ss << "now = " << irreg_progress.get_date() << "   ";
            ss << "finish = " << irreg_progress.get_date_fin();
            date_out = ss.str();

            // -----------------------------------
            // c2. Terminal Output
            // -----------------------------------
            std::cout << date_out << std::endl << progress::separator() << std::endl;
            ofs << date_out << std::endl << progress::separator() << std::endl;

        }

    }

    // Post Processing
    ofs.close();

    // End Processing
    return;

}
