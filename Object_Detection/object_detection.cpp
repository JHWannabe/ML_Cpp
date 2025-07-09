#include "object_detection.h"

int mainObjectDetection(int argc, const char* argv[], std::string file_path) {

    if (!std::filesystem::exists(file_path))
    {
        return 1;
    }
    // first, create a file instance
    mINI::INIFile file(file_path);
    // next, create a structure that will hold data
    mINI::INIStructure ini;

    // now we can read the file
    if (!file.read(ini))
        return 1;

    // (2) Select Device
    torch::Device device = Set_Device(ini);
    std::cout << "using device = " << device << std::endl;

    // (3) Set Seed
    if (stringToBool(ini["General"]["seed_random"])) {
        std::random_device rd;
        std::srand(rd());
        torch::manual_seed(std::rand());
        torch::globalContext().setDeterministicCuDNN(false);
        torch::globalContext().setBenchmarkCuDNN(true);
    }
    else {
        std::srand(std::stoi(ini["General"]["seed"]));
        torch::manual_seed(std::rand());
        torch::globalContext().setDeterministicCuDNN(true);
        torch::globalContext().setBenchmarkCuDNN(false);
    }

    // (4) Set Transforms
    std::vector<transforms_Compose> transformBB;
    if (stringToBool(ini["Training"]["augmentation"])) {
        transformBB.push_back(
            YOLOAugmentation(  // apply "jitter", "flip", "scale", "blur", "brightness", "hue", "saturation", "shift", "crop"
                std::stod(ini["Training"]["jitter"]),
                std::stod(ini["Training"]["flip_rate"]),
                std::stod(ini["Training"]["scale_rate"]),
                std::stod(ini["Training"]["blur_rate"]),
                std::stod(ini["Training"]["brightness_rate"]),
                std::stod(ini["Training"]["hue_rate"]),
                std::stod(ini["Training"]["saturation_rate"]),
                std::stod(ini["Training"]["shift_rate"]),
                std::stod(ini["Training"]["crop_rate"])
            )
        );
    }
    /*************************************************************************/
    std::vector<transforms_Compose> transformI{
          transforms_Resize(cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"])), cv::INTER_LINEAR), // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
          transforms_ToTensor(),                                                                                  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
    };
    if (std::stol(ini["General"]["input_channel"]) == 1) {
        transformI.insert(transformI.begin(), transforms_Grayscale(1));
    }
    /*************************************************************************/
    std::vector<transforms_Compose> transformD{
        transforms_ToTensor()  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
    };
    if (std::stol(ini["General"]["input_channel"]) == 1) {
        transformD.insert(transformD.begin(), transforms_Grayscale(1));
    }

    // (5) Define Network
    YOLOv3 model(ini);
    model->to(device);

    // (6) Make Directories
    std::string dir = "../Object_Detection/checkpoints/" + ini["General"]["dataset"];
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(ini, model, "YOLOv3");

    // (8) Set Class Names and Configs
    std::vector<std::string> class_names = Set_Class_Names(ini["General"]["class_list"], std::stol(ini["General"]["class_num"]));
    std::vector<std::vector<std::tuple<float, float>>> anchors = Set_Anchors(ini["General"]["anchor_list"], std::stol(ini["General"]["scales"]), std::stol(ini["General"]["num_anchor"]));
    size_t resize_step_max;
    std::vector<std::tuple<long int, long int>> resizes = Set_Resizes(ini["General"]["resize_list"], resize_step_max);

    // (9.1) Training Phase
    if (stringToBool(ini["Training"]["train"])) {
        Set_Options(ini, argc, argv, "train");
        train(ini, device, model, transformBB, transformI, class_names, anchors, resizes, resize_step_max);
    }

    // (9.2) Test Phase
    if (stringToBool(ini["Test"]["test"])) {
        Set_Options(ini, argc, argv, "test");
        test(ini, device, model, transformI, class_names, anchors);
    }

    // (9.3) Detection Phase
    if (stringToBool(ini["Detection"]["detect"])) {
        Set_Options(ini, argc, argv, "detect");
        detect(ini, device, model, transformI, transformD, class_names, anchors);
    }

    // (9.4) Demo Phase
    if (stringToBool(ini["Demo"]["demo"])) {
        Set_Options(ini, argc, argv, "demo");
        demo(ini, device, model, transformI, transformD, class_names, anchors);
    }

    // End Processing
    return 0;

}

torch::Device Set_Device(mINI::INIStructure& ini) {

    // (1) GPU Type
    int gpu_id = std::stoi(ini["General"]["gpu_id"]);
    if (torch::cuda::is_available() && gpu_id >= 0) {
        torch::Device device(torch::kCUDA, gpu_id);
        return device;
    }

    // (2) CPU Type
    torch::Device device(torch::kCPU);
    return device;

}

void Set_Model_Params(mINI::INIStructure& ini, YOLOv3& model, const std::string name) {

    // (1) Make Directory
    std::string dir = "../Object_Detection/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"] + "/model_params/";
    fs::create_directories(dir);

    // (2.1) File Open
    std::string fname = dir + name + ".txt";
    std::ofstream ofs(fname);

    // (2.2) Calculation of Parameters
    size_t num_params = 0;
    for (auto param : model->parameters()) {
        num_params += param.numel();
    }
    ofs << "Total number of parameters : " << (float)num_params / 1e6f << "M" << std::endl << std::endl;
    ofs << model << std::endl;

    // (2.3) File Close
    ofs.close();

    // End Processing
    return;

}

std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num) {

    // (1) Memory Allocation
    std::vector<std::string> class_names = std::vector<std::string>(class_num);

    // (2) Get Class Names
    std::string class_name;
    std::ifstream ifs(path, std::ios::in);
    for (size_t i = 0; i < class_num; i++) {
        if (!getline(ifs, class_name)) {
            std::cerr << "Error : The number of classes does not match the number of lines in the class name file." << std::endl;
            std::exit(1);
        }
        class_names.at(i) = class_name;
    }
    ifs.close();

    // End Processing
    return class_names;

}

std::vector<std::vector<std::tuple<float, float>>> Set_Anchors(const std::string path, const size_t scales, const size_t na) {

    // (1) Memory Allocation
    std::vector<std::vector<std::tuple<float, float>>> anchors = std::vector<std::vector<std::tuple<float, float>>>(scales);
    for (size_t i = 0; i < scales; i++) {
        anchors.at(i) = std::vector<std::tuple<float, float>>(na);
    }

    // (2) Get Anchors
    float pw, ph;
    std::string line;
    std::ifstream ifs(path, std::ios::in);
    for (size_t i = 0; i < scales; i++) {
        for (size_t j = 0; j < na; j++) {
            if (!getline(ifs, line)) {
                std::cerr << "Error : The number of anchors does not match the number of lines in the anchor file." << std::endl;
                std::exit(1);
            }
            std::istringstream iss(line);
            iss >> pw;
            iss >> ph;
            anchors.at(i).at(j) = { pw, ph };
        }
    }
    ifs.close();

    // End Processing
    return anchors;

}

std::vector<std::tuple<long int, long int>> Set_Resizes(const std::string path, size_t& resize_step_max) {

    // (1) Memory Allocation
    std::vector<std::tuple<long int, long int>> resizes;

    // (2) Get Resizes
    size_t num;
    long int width, height;
    std::string line;
    std::ifstream ifs(path, std::ios::in);
    /**********************************************************************/
    for (size_t i = 0; i < 2; i++) {
        if (!getline(ifs, line)) {
            std::cerr << "Error : The number of configs and resizes does not match the number of lines in the resize file." << std::endl;
            std::exit(1);
        }
        std::istringstream iss(line);
        if (i == 0) {
            iss >> num;
        }
        else {
            iss >> resize_step_max;
        }
    }
    /**********************************************************************/
    resizes = std::vector<std::tuple<long int, long int>>(num);
    for (size_t i = 0; i < num; i++) {
        if (!getline(ifs, line)) {
            std::cerr << "Error : The number of configs and resizes does not match the number of lines in the resize file." << std::endl;
            std::exit(1);
        }
        std::istringstream iss(line);
        iss >> width;
        iss >> height;
        resizes.at(i) = { width, height };
    }
    ifs.close();

    // End Processing
    return resizes;

}

void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], const std::string mode) {

    // (1) Make Directory
    std::string dir = "../Object_Detection/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"] + "/options/";
    fs::create_directories(dir);

    // (3.1) File Open
    std::string fname = dir + mode + ".txt";
    std::ofstream ofs(fname, std::ios::app);

    // (3.2) Arguments Output
    ofs << "--------------------------------------------" << std::endl;
    ofs << "Command Line Arguments:" << std::endl;
    for (int i = 1; i < argc; i++) {
        if (i % 2 == 1) {
            ofs << "  " << argv[i] << '\t' << std::flush;
        }
        else {
            ofs << argv[i] << std::endl;
        }
    }

    // (3.3) File Close
    ofs.close();

    // End Processing
    return;

}

bool stringToBool(const std::string& str)
{
    // Create a copy of the string to convert to lowercase for comparison
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    // Convert various values corresponding to "true" to true
    if (lowerStr == "true" || lowerStr == "1") {
        return true;
    }
    else if (lowerStr == "false" || lowerStr == "0") {
        return false;
    }

    // Exception handling, return false or throw exception for other values
    throw std::invalid_argument("Invalid boolean string value");
}

void train(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transformBB, std::vector<transforms_Compose>& transformI, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors, const std::vector<std::tuple<long int, long int>> resizes, const size_t resize_step_max) {

    // Set training and validation data loader options
    bool train_shuffle = ini["Training"]["train_shuffle"] == "true";  // Shuffle training dataset
    size_t train_workers = 4;  // Number of workers for training data loading
    bool valid_shuffle = true;  // Shuffle validation dataset
    size_t valid_workers = 4;  // Number of workers for validation data loading
    size_t save_sample_iter = 50;  // Frequency to save sample images
    std::string_view extension = "jpg";  // Extension for saved sample images
    std::pair<float, float> output_range = { 0.0, 1.0 };  // Output image value range

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch, iter;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    size_t resize_step;
    size_t idx;
    long int width, height;
    float loss_f, loss_coord_xy_f, loss_coord_wh_f, loss_obj_f, loss_noobj_f, loss_class_f;
    float lr_init, lr_base, lr_decay1, lr_decay2;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string input_dir, output_dir;
    std::string valid_input_dir, valid_output_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> urand;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image;
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    std::vector<torch::Tensor> output, output_one;
    cv::Mat sample;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::vector<transforms_Compose> null;
    datasets::ImageFolderBBWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderBBWithPaths dataloader, valid_dataloader;
    std::vector<visualizer::graph> train_loss;
    std::vector<visualizer::graph> valid_loss;
    progress::display* show_progress;
    progress::irregular irreg_progress;

    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    cv::setNumThreads(0);  // Disable OpenCV multithreading for consistent processing

    // (1) Load training dataset
    input_dir = "../Object_Detection/datasets/" + ini["General"]["dataset"] + "/" + ini["Training"]["train_in_dir"];
    output_dir = "../Object_Detection/datasets/" + ini["General"]["dataset"] + "/" + ini["Training"]["train_out_dir"];
    dataset = datasets::ImageFolderBBWithPaths(input_dir, output_dir, transformBB, transformI);
    dataloader = DataLoader::ImageFolderBBWithPaths(dataset, std::stol(ini["Training"]["batch_size"]), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Load validation dataset if enabled
    if (stringToBool(ini["Validation"]["valid"])) {
        valid_input_dir = "../Object_Detection/datasets/" + ini["General"]["dataset"] + "/" + ini["Validation"]["valid_in_dir"];
        valid_output_dir = "../Object_Detection/datasets/" + ini["General"]["dataset"] + "/" + ini["Validation"]["valid_out_dir"];
        valid_dataset = datasets::ImageFolderBBWithPaths(valid_input_dir, valid_output_dir, null, transformI);
        valid_dataloader = DataLoader::ImageFolderBBWithPaths(valid_dataset, std::stol(ini["Validation"]["valid_batch_size"]), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set optimizer (SGD)
    using Optimizer = torch::optim::SGD;
    using OptimizerOptions = torch::optim::SGDOptions;
    auto optimizer = Optimizer(model->parameters(), OptimizerOptions(std::stof(ini["Network"]["lr_init"])).momentum(std::stof(ini["Network"]["momentum"])).weight_decay(std::stof(ini["Network"]["weight_decay"])));

    // (4) Set loss function
    auto criterion = Loss(anchors, (long int)std::stol(ini["General"]["class_num"]), std::stof(ini["General"]["ignore_thresh"]));

    // (5) Set detector for visualization
    auto detector = YOLODetector(anchors, (long int)std::stol(ini["General"]["class_num"]), std::stof(ini["General"]["prob_thresh"]), std::stof(ini["General"]["nms_thresh"]));
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (6) Create necessary directories for checkpoints, logs, and samples
    checkpoint_dir = "../Object_Detection/checkpoints/" + ini["General"]["dataset"];
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/samples";  fs::create_directories(save_images_dir);

    // (7) Initialize loss graphs for training and validation
    path = checkpoint_dir + "/graph";
    train_loss = std::vector<visualizer::graph>(6);
    train_loss.at(0) = visualizer::graph(path, /*gname_=*/"train_loss_all", /*label_=*/{ "Total" });
    train_loss.at(1) = visualizer::graph(path, /*gname_=*/"train_loss_coord_center", /*label_=*/{ "Coordinate(center)" });
    train_loss.at(2) = visualizer::graph(path, /*gname_=*/"train_loss_coord_range", /*label_=*/{ "Coordinate(range)" });
    train_loss.at(3) = visualizer::graph(path, /*gname_=*/"train_loss_conf_obj", /*label_=*/{ "Confidence(object)" });
    train_loss.at(4) = visualizer::graph(path, /*gname_=*/"train_loss_conf_noobj", /*label_=*/{ "Confidence(no-object)" });
    train_loss.at(5) = visualizer::graph(path, /*gname_=*/"train_loss_class", /*label_=*/{ "Class" });
    if (stringToBool(ini["Validation"]["valid"])) {
        valid_loss = std::vector<visualizer::graph>(6);
        valid_loss.at(0) = visualizer::graph(path, /*gname_=*/"valid_loss_all", /*label_=*/{ "Total" });
        valid_loss.at(1) = visualizer::graph(path, /*gname_=*/"valid_loss_coord_center", /*label_=*/{ "Coordinate(center)" });
        valid_loss.at(2) = visualizer::graph(path, /*gname_=*/"valid_loss_coord_range", /*label_=*/{ "Coordinate(range)" });
        valid_loss.at(3) = visualizer::graph(path, /*gname_=*/"valid_loss_conf_obj", /*label_=*/{ "Confidence(object)" });
        valid_loss.at(4) = visualizer::graph(path, /*gname_=*/"valid_loss_conf_noobj", /*label_=*/{ "Confidence(no-object)" });
        valid_loss.at(5) = visualizer::graph(path, /*gname_=*/"valid_loss_class", /*label_=*/{ "Class" });
    }

    // (8) Load weights or initialize model
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

            // Print parameter names in traced model
            for (const auto& pair : traced_model.named_parameters()) {
                std::cout << pair.name << std::endl;
            }

            // Print parameter names in current model
            for (const auto& pair : model->named_parameters()) {
                std::cout << pair.key() << std::endl;
            }

            // Copy parameters from traced model to current model
            for (const auto& pair : traced_model.named_parameters()) {
                auto param_name = pair.name;
                auto param_tensor = pair.value;

                // Find parameter in current model by name
                auto params = model->named_parameters();
                auto it = std::find_if(params.begin(), params.end(),
                    [&](const auto& p) { return p.key() == param_name; });

                if (it != params.end()) {
                    // Check shape match
                    if (it->value().sizes() != param_tensor.sizes()) {
                        std::cerr << "Shape mismatch for " << param_name
                            << ": model size: " << it->value().sizes()
                            << ", traced size: " << param_tensor.sizes() << std::endl;
                        continue;
                    }

                    // Copy parameter
                    try {
                        torch::NoGradGuard no_grad;  // Disable gradient tracking
                        it->value().copy_(param_tensor.detach());  // In-place copy
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
        else
        {
            model->apply(weights_init); // Initialize model weights
            ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
            if (stringToBool(ini["Validation"]["valid"])) {
                init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
                init.close();
            }
        }
        start_epoch = 0;
    }
    else {
        // Load model and optimizer from checkpoint
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

    // (9) Print current date and training header
    date = progress::current_date();
    date = progress::separator_center("Train Loss (" + date + ")");
    std::cout << std::endl << std::endl << date << std::endl;
    ofs << date << std::endl;

    // -----------------------------------
    // a2. Training Model
    // -----------------------------------

    // (1) Set training parameters
    start_epoch++;
    total_iter = dataloader.get_count_max();
    mt.seed(std::rand());
    urand = std::uniform_int_distribution<size_t>(/*min=*/0, /*max=*/resizes.size() - 1);
    resize_step = 0;
    idx = urand(mt);
    width = std::get<0>(resizes.at(idx));
    height = std::get<1>(resizes.at(idx));
    total_epoch = std::stol(ini["Training"]["epochs"]);
    lr_init = std::stof(ini["Network"]["lr_init"]);
    lr_base = std::stof(ini["Network"]["lr_base"]);
    lr_decay1 = std::stof(ini["Network"]["lr_decay1"]);
    lr_decay2 = std::stof(ini["Network"]["lr_decay2"]);

    // (2) Training loop for each epoch
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++) {

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{ epoch, total_epoch }, /*loss_=*/{ "coord_xy", "coord_wh", "conf_o", "conf_x", "class", "W", "H" });
        std::cout << std::endl;

        // Print learning rate for current epoch
        std::cout << "Epoch [" << epoch << "] Learning Rate: "
            << optimizer.param_groups()[0].options().get_lr() << std::endl;

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)) {

            image = std::get<0>(mini_batch).to(device);  // {N,C,H,W} (images)
            label = std::get<1>(mini_batch);  // {N, ({BB_n}, {BB_n,4}) } (annotations)

            // (c1) Update learning rate
            Update_LR<Optimizer, OptimizerOptions>(optimizer, lr_init, lr_base, lr_decay1, lr_decay2, epoch, (float)show_progress->get_iters() / (float)total_iter);

            // (c2) Randomly resize images for multi-scale training
            resize_step++;
            if (resize_step > resize_step_max) {
                resize_step = 1;
                idx = urand(mt);
                width = std::get<0>(resizes.at(idx));
                height = std::get<1>(resizes.at(idx));
            }
            image = F::interpolate(image, F::InterpolateFuncOptions().size(std::vector<int64_t>({ height, width })).mode(torch::kBilinear).align_corners(false));

            // (c3) Forward pass and compute loss
            output = model->forward(image);  // {N,C,H,W} ===> {S,{N,G,G,A*(CN+5)}}
            losses = criterion(output, label, { (float)width, (float)height });
            loss_coord_xy = std::get<0>(losses) * std::stof(ini["Network"]["Lambda_coord"]);
            loss_coord_wh = std::get<1>(losses) * std::stof(ini["Network"]["Lambda_coord"]);
            loss_obj = std::get<2>(losses) * std::stof(ini["Network"]["Lambda_object"]);
            loss_noobj = std::get<3>(losses) * std::stof(ini["Network"]["Lambda_noobject"]);
            loss_class = std::get<4>(losses) * std::stof(ini["Network"]["Lambda_class"]);
            loss = loss_coord_xy + loss_coord_wh + loss_obj + loss_noobj + loss_class;
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // (c4) Record loss for this iteration
            show_progress->increment(/*loss_value=*/{ loss_coord_xy.item<float>(), loss_coord_wh.item<float>(), loss_obj.item<float>(), loss_noobj.item<float>(), loss_class.item<float>(), (float)width, (float)height });
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "coord_xy:" << loss_coord_xy.item<float>() << "(ave:" << show_progress->get_ave(0) << ") " << std::flush;
            ofs << "coord_wh:" << loss_coord_wh.item<float>() << "(ave:" << show_progress->get_ave(1) << ") " << std::flush;
            ofs << "conf_o:" << loss_obj.item<float>() << "(ave:" << show_progress->get_ave(2) << ") " << std::flush;
            ofs << "conf_x:" << loss_noobj.item<float>() << "(ave:" << show_progress->get_ave(3) << ") " << std::flush;
            ofs << "class:" << loss_class.item<float>() << "(ave:" << show_progress->get_ave(4) << ") " << std::flush;
            ofs << "(W,H):" << "(" << width << "," << height << ")" << std::endl;

            // (c5) Save sample images at specified intervals
            iter = show_progress->get_iters();
            if (iter % save_sample_iter == 1) {
                ss.str(""); ss.clear(std::stringstream::goodbit);
                ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
                /*************************************************************************/
                output_one = std::vector<torch::Tensor>(output.size());
                for (size_t i = 0; i < output_one.size(); i++) {
                    output_one.at(i) = output.at(i)[0];
                }
                detect_result = detector(output_one, { (float)width, (float)height });
                /*************************************************************************/
                sample = visualizer::draw_detections_des(image[0].detach(), { std::get<0>(detect_result), std::get<1>(detect_result) }, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);
                cv::imwrite(ss.str(), sample);
            }

        }

        // -----------------------------------
        // b2. Record average loss for this epoch
        // -----------------------------------
        loss_f = show_progress->get_ave(0) + show_progress->get_ave(1) + show_progress->get_ave(2) + show_progress->get_ave(3) + show_progress->get_ave(4);
        loss_coord_xy_f = show_progress->get_ave(0);
        loss_coord_wh_f = show_progress->get_ave(1);
        loss_obj_f = show_progress->get_ave(2);
        loss_noobj_f = show_progress->get_ave(3);
        loss_class_f = show_progress->get_ave(4);
        train_loss.at(0).plot(/*base=*/epoch, /*value=*/{ loss_f });
        train_loss.at(1).plot(/*base=*/epoch, /*value=*/{ loss_coord_xy_f });
        train_loss.at(2).plot(/*base=*/epoch, /*value=*/{ loss_coord_wh_f });
        train_loss.at(3).plot(/*base=*/epoch, /*value=*/{ loss_obj_f });
        train_loss.at(4).plot(/*base=*/epoch, /*value=*/{ loss_noobj_f });
        train_loss.at(5).plot(/*base=*/epoch, /*value=*/{ loss_class_f });

        // -----------------------------------
        // b3. Save sample image for this epoch
        // -----------------------------------
        ss.str(""); ss.clear(std::stringstream::goodbit);
        ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        /*************************************************************************/
        output_one = std::vector<torch::Tensor>(output.size());
        for (size_t i = 0; i < output_one.size(); i++) {
            output_one.at(i) = output.at(i)[0];
        }
        detect_result = detector(output_one, { (float)width, (float)height });
        /*************************************************************************/
        sample = visualizer::draw_detections_des(image[0].detach(), { std::get<0>(detect_result), std::get<1>(detect_result) }, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);
        cv::imwrite(ss.str(), sample);
        delete show_progress;

        // -----------------------------------
        // b4. Run validation if enabled and at specified frequency
        // -----------------------------------
        if (stringToBool(ini["Validation"]["valid"]) && ((epoch - 1) % std::stol(ini["Validation"]["valid_freq"]) == 0)) {
            valid(ini, valid_dataloader, device, criterion, model, class_names, epoch, valid_loss);
        }

        // -----------------------------------
        // b5. Save model and optimizer checkpoints
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
        // b6. Print elapsed and estimated remaining time every 10 epochs
        // -----------------------------------
        if (epoch % 10 == 0) {

            // (c1) Prepare output string for time information
            ss.str(""); ss.clear(std::stringstream::goodbit);
            irreg_progress.nab(epoch);
            ss << "elapsed = " << irreg_progress.get_elap() << '(' << irreg_progress.get_sec_per() << "sec/epoch)   ";
            ss << "remaining = " << irreg_progress.get_rem() << "   ";
            ss << "now = " << irreg_progress.get_date() << "   ";
            ss << "finish = " << irreg_progress.get_date_fin();
            date_out = ss.str();

            // (c2) Print time information to terminal and log
            std::cout << date_out << std::endl << progress::separator() << std::endl;
            ofs << date_out << std::endl << progress::separator() << std::endl;

        }

    }

    // Close log file after training
    ofs.close();

    // End Processing
    return;

}

template <typename Optimizer, typename OptimizerOptions>
void Update_LR(Optimizer& optimizer, const float lr_init, const float lr_base, const float lr_decay1, const float lr_decay2, const size_t epoch, const float burnin_base, const float burnin_exp) {

    float lr;

    if (epoch == 1) {
        lr = lr_init + (lr_base - lr_init) * std::pow(burnin_base, burnin_exp);
    }
    else if (epoch == 5) {
        lr = lr_base;
    }
    else if (epoch == 60) {
        lr = lr_decay1;
    }
    else if (epoch == 90) {
        lr = lr_decay2;
    }
    else {
        return;
    }

    for (auto& param_group : optimizer.param_groups()) {
        if (param_group.has_options()) {
            auto& options = (OptimizerOptions&)(param_group.options());
            options.lr(lr);
        }
    }

    return;

}

void valid(mINI::INIStructure& ini, DataLoader::ImageFolderBBWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, YOLOv3& model, const std::vector<std::string> class_names, const size_t epoch, std::vector<visualizer::graph>& writer) {

    // (0) Initialization and Declaration
    size_t iteration;
    float ave_loss_coord_xy, total_loss_coord_xy;
    float ave_loss_coord_wh, total_loss_coord_wh;
    float ave_loss_obj, total_loss_obj;
    float ave_loss_noobj, total_loss_noobj;
    float ave_loss_class, total_loss_class;
    float ave_loss_all;
    std::ofstream ofs;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor image;
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    std::vector<torch::Tensor> output;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss_coord_xy = 0.0; total_loss_coord_wh = 0.0; total_loss_obj = 0.0; total_loss_noobj = 0.0; total_loss_class = 0.0;
    while (valid_dataloader(mini_batch)) {

        image = std::get<0>(mini_batch).to(device);  // {N,C,H,W} (images)
        label = std::get<1>(mini_batch);  // {N, ({BB_n}, {BB_n,4}) } (annotations)

        output = model->forward(image);  // {N,C,H,W} ===> {S,{N,G,G,FF}}
        losses = criterion(output, label, { (float)image.size(3), (float)image.size(2) });

        loss_coord_xy = std::get<0>(losses) * std::stof(ini["Network"]["Lambda_coord"]);
        loss_coord_wh = std::get<1>(losses) * static_cast<float>(std::stol(ini["Network"]["Lambda_coord"]));
        loss_obj = std::get<2>(losses) * std::stof(ini["Network"]["Lambda_object"]);
        loss_noobj = std::get<3>(losses) * std::stof(ini["Network"]["Lambda_noobject"]);
        loss_class = std::get<4>(losses) * std::stof(ini["Network"]["Lambda_class"]);

        total_loss_coord_xy += loss_coord_xy.item<float>();
        total_loss_coord_wh += loss_coord_wh.item<float>();
        total_loss_obj += loss_obj.item<float>();
        total_loss_noobj += loss_noobj.item<float>();
        total_loss_class += loss_class.item<float>();

        iteration++;

    }

    // (2) Calculate Average Loss
    ave_loss_coord_xy = total_loss_coord_xy / (float)iteration;
    ave_loss_coord_wh = total_loss_coord_wh / (float)iteration;
    ave_loss_obj = total_loss_obj / (float)iteration;
    ave_loss_noobj = total_loss_noobj / (float)iteration;
    ave_loss_class = total_loss_class / (float)iteration;
    ave_loss_all = ave_loss_coord_xy + ave_loss_coord_wh + ave_loss_obj + ave_loss_noobj + ave_loss_class;

    // (3.1) Record Loss (Log)
    ofs.open("checkpoints/" + ini["General"]["dataset"] + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << std::stol(ini["Training"]["epochs"]) << ' ' << std::flush;
    ofs << "coord_xy:" << ave_loss_coord_xy << ' ' << std::flush;
    ofs << "coord_wh:" << ave_loss_coord_wh << ' ' << std::flush;
    ofs << "conf_o:" << ave_loss_obj << ' ' << std::flush;
    ofs << "conf_x:" << ave_loss_noobj << ' ' << std::flush;
    ofs << "class:" << ave_loss_class << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.at(0).plot(/*base=*/epoch, /*value=*/{ ave_loss_all });
    writer.at(1).plot(/*base=*/epoch, /*value=*/{ ave_loss_coord_xy });
    writer.at(2).plot(/*base=*/epoch, /*value=*/{ ave_loss_coord_wh });
    writer.at(3).plot(/*base=*/epoch, /*value=*/{ ave_loss_obj });
    writer.at(4).plot(/*base=*/epoch, /*value=*/{ ave_loss_noobj });
    writer.at(5).plot(/*base=*/epoch, /*value=*/{ ave_loss_class });

    // End Processing
    return;

}

void test(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transform, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors) {

    // (0) Initialization and Declaration
    float ave_loss_coord_xy, ave_loss_coord_wh, ave_loss_obj, ave_loss_noobj, ave_loss_class;
    double seconds, ave_time;
    std::string path, result_dir;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, std::vector<std::tuple<torch::Tensor, torch::Tensor>>, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor image;
    torch::Tensor loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_class;
    std::vector<torch::Tensor> output;
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> label;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> losses;
    datasets::ImageFolderBBWithPaths dataset;
    DataLoader::ImageFolderBBWithPaths dataloader;
    std::vector<transforms_Compose> null;

    // (1) Get Test Dataset
    input_dir = "../Object_Detection/datasets/" + ini["General"]["dataset"] + '/' + ini["Test"]["test_in_dir"];
    output_dir = "../Object_Detection/datasets/" + ini["General"]["dataset"] + '/' + ini["Test"]["test_out_dir"];
    dataset = datasets::ImageFolderBBWithPaths(input_dir, output_dir, null, transform);
    dataloader = DataLoader::ImageFolderBBWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "../Object_Detection/checkpoints/" + ini["General"]["dataset"] + "/models/epoch_" + ini["Test"]["test_load_epoch"] + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = Loss(anchors, (long int)std::stol(ini["General"]["class_num"]), std::stof(ini["General"]["ignore_thresh"]));

    // (4) Initialization of Value
    ave_loss_coord_xy = 0.0;
    ave_loss_coord_wh = 0.0;
    ave_loss_obj = 0.0;
    ave_loss_noobj = 0.0;
    ave_loss_class = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = ini["Test"]["test_result_dir"];  fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)) {

        image = std::get<0>(data).to(device);
        label = std::get<1>(data);

        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();

        output = model->forward(image);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

        losses = criterion(output, label, { (float)image.size(3), (float)image.size(2) });
        loss_coord_xy = std::get<0>(losses) * std::stof(ini["Network"]["Lambda_coord"]);
        loss_coord_wh = std::get<1>(losses) * std::stof(ini["Network"]["Lambda_coord"]);
        loss_obj = std::get<2>(losses) * std::stof(ini["Network"]["Lambda_object"]);
        loss_noobj = std::get<3>(losses) * std::stof(ini["Network"]["Lambda_noobject"]);
        loss_class = std::get<4>(losses) * std::stof(ini["Network"]["Lambda_class"]);

        ave_loss_coord_xy += loss_coord_xy.item<float>();
        ave_loss_coord_wh += loss_coord_wh.item<float>();
        ave_loss_obj += loss_obj.item<float>();
        ave_loss_noobj += loss_noobj.item<float>();
        ave_loss_class += loss_class.item<float>();
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> coord_xy:" << loss_coord_xy.item<float>() << " coord_wh:" << loss_coord_wh.item<float>() << " conf_o:" << loss_obj.item<float>() << " conf_x:" << loss_noobj.item<float>() << " class:" << loss_class.item<float>() << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> coord_xy:" << loss_coord_xy.item<float>() << " coord_wh:" << loss_coord_wh.item<float>() << " conf_o:" << loss_obj.item<float>() << " conf_x:" << loss_noobj.item<float>() << " class:" << loss_class.item<float>() << std::endl;

    }

    // (6) Calculate Average
    ave_loss_coord_xy = ave_loss_coord_xy / (float)dataset.size();
    ave_loss_coord_wh = ave_loss_coord_wh / (float)dataset.size();
    ave_loss_obj = ave_loss_obj / (float)dataset.size();
    ave_loss_noobj = ave_loss_noobj / (float)dataset.size();
    ave_loss_class = ave_loss_class / (float)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> coord_xy:" << ave_loss_coord_xy << " coord_wh:" << ave_loss_coord_wh << " conf_o:" << ave_loss_obj << " conf_x:" << ave_loss_noobj << " class:" << ave_loss_class << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> coord_xy:" << ave_loss_coord_xy << " coord_wh:" << ave_loss_coord_wh << " conf_o:" << ave_loss_obj << " conf_x:" << ave_loss_noobj << " class:" << ave_loss_class << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}

void detect(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformD, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors) {

    constexpr std::pair<float, float> output_range = { 0.0, 1.0 };  // range of the value in output images

    // (0) Initialization and Declaration
    size_t BB_n;
    float prob;
    std::string path, result_dir, fname;
    std::string dataroot;
    std::string class_name;
    std::stringstream ss;
    std::ofstream ofs;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
    torch::Tensor imageI, imageD;
    torch::Tensor ids, coords, probs;
    std::vector<torch::Tensor> output, output_one;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    cv::Mat imageO;
    datasets::ImageFolderPairWithPaths dataset;
    DataLoader::ImageFolderPairWithPaths dataloader;

    // (1) Get Detection Dataset
    dataroot = "./datasets/" + ini["General"]["dataset"] + '/' + ini["Detection"]["detect_dir"];
    dataset = datasets::ImageFolderPairWithPaths(dataroot, dataroot, transformI, transformD);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total detect images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "checkpoints/" + ini["General"]["dataset"] + "/models/epoch_" + ini["Detection"]["detect_load_epoch"] + ".pth";
    torch::load(model, path, device);

    // (3) Set Detector
    auto detector = YOLODetector(anchors, (long int)std::stol(ini["General"]["class_num"]), std::stof(ini["General"]["prob_thresh"]), std::stof(ini["General"]["nms_thresh"]));
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (4) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = ini["Detection"]["detect_result_dir"];  fs::create_directories(result_dir);
    ofs.open(result_dir + "/detect.txt", std::ios::out);
    while (dataloader(data)) {

        // (4.1) Get data
        imageI = std::get<0>(data).to(device);  // {1,C,H,W} (image for input)
        imageD = std::get<1>(data);  // {1,3,H_D,W_D} (image for detection)

        // (4.2) Inference and Detection
        output = model->forward(imageI);  // {1,C,H,W} ===> {S,{1,G,G,FF}}
        /*************************************************************************/
        output_one = std::vector<torch::Tensor>(output.size());
        for (size_t i = 0; i < output_one.size(); i++) {
            output_one.at(i) = output.at(i)[0];
        }
        detect_result = detector(output_one, { (float)imageI.size(3), (float)imageI.size(2) });  // output_one{S,{G,G,FF}} ===> detect_result{ (ids{BB_n}, coords{BB_n,4}, probs{BB_n}) }
        /*************************************************************************/
        ids = std::get<0>(detect_result);  // ids{BB_n}
        coords = std::get<1>(detect_result);  // coords{BB_n,4}
        probs = std::get<2>(detect_result);  // probs{BB_n}
        imageO = visualizer::draw_detections_des(imageD[0].detach(), { ids, coords }, probs, class_names, label_palette, /*range=*/output_range);

        // (4.3) Save image
        fname = result_dir + '/' + std::get<2>(data).at(0);  // {1,C,H,W} ===> {1,G,G,FF}
        cv::imwrite(fname, imageO);

        // (4.4) Write detection result
        BB_n = ids.size(0);
        std::cout << '<' << std::get<2>(data).at(0) << "> " << BB_n << " { " << std::flush;
        ofs << '<' << std::get<2>(data).at(0) << "> " << BB_n << " { " << std::flush;
        for (size_t i = 0; i < BB_n; i++) {
            class_name = class_names.at(ids[i].item<int64_t>());
            prob = probs[i].item<float>() * 100.0;
            ss.str(""); ss.clear(std::stringstream::goodbit);
            ss << std::fixed << std::setprecision(1) << prob;
            std::cout << class_name << ":" << ss.str() << "% " << std::flush;
            ofs << class_name << ":" << ss.str() << "% " << std::flush;
        }
        std::cout << "}" << std::endl;
        ofs << "}" << std::endl;

    }

    // Post Processing
    ofs.close();

    // End Processing
    return;

}

void demo(mINI::INIStructure& ini, torch::Device& device, YOLOv3& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformD, const std::vector<std::string> class_names, const std::vector<std::vector<std::tuple<float, float>>> anchors) {

    constexpr double alpha = 0.1;  // current importance of moving average for calculating FPS
    constexpr std::pair<float, float> output_range = { 0.0, 1.0 };  // range of the value in output images

    // (0) Initialization and Declaration
    int key;
    bool flag;
    double seconds, seconds_est, fps;
    std::string path;
    std::stringstream ss1, ss2;
    std::chrono::system_clock::time_point start, end;
    cv::Mat BGR, RGB;
    cv::Mat imageD, imageI, imageO;
    torch::Tensor tensorD, tensorI;
    std::vector<torch::Tensor> tensorO, tensorO_one;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> detect_result;
    cv::VideoCapture cap;

    // (1) Get Model
    path = "checkpoints/" + ini["General"]["dataset"] + "/models/epoch_" + ini["Demo"]["demo_load_epoch"] + ".pth";
    torch::load(model, path, device);

    // (2) Set Detector
    auto detector = YOLODetector(anchors, (long int)std::stol(ini["General"]["class_num"]), std::stof(ini["General"]["prob_thresh"]), std::stof(ini["Network"]["nms_thresh"]));
    std::vector<std::tuple<unsigned char, unsigned char, unsigned char>> label_palette = detector.get_label_palette();

    // (3) Set Camera Device
    if (ini["Demo"]["movie"] == "") {
        cap.open(std::stol(ini["Demo"]["cam_num"]));
        if (!cap.isOpened()) {
            std::cerr << "Error : Couldn't open the camera '" << std::stol(ini["Demo"]["cam_num"]) << "'." << std::endl;
            std::exit(1);
        }
        else {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, std::stol(ini["Demo"]["window_w"]));
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, std::stol(ini["Demo"]["window_h"]));
        }
    }
    else {
        cap.open(ini["Demo"]["movie"]);
        if (!cap.isOpened()) {
            std::cerr << "Error : Couldn't open the movie '" << ini["Demo"]["movie"] << "'." << std::endl;
            std::exit(1);
        }
    }

    // (4) Show Key Information
    std::cout << std::endl;
    std::cout << "<Key Information>" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "| key |        action        |" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << "|  q  | Stop the camera.     |" << std::endl;
    std::cout << "------------------------------" << std::endl;
    std::cout << std::endl;

    // (5) Demo
    torch::NoGradGuard no_grad;
    model->eval();
    flag = true;
    fps = 0.0;
    start = std::chrono::system_clock::now();
    while (cap.read(BGR)) {

        cv::cvtColor(BGR, RGB, cv::COLOR_BGR2RGB);  // {0,1,2} = {B,G,R} ===> {0,1,2} = {R,G,B}

        // (5.1) Set image for input
        RGB.copyTo(imageI);
        tensorI = transforms::apply(transformI, imageI).unsqueeze(/*dim=*/0).to(device);  // imageI{H_D,W_D,C} ==={Resize,ToTensor,etc.}===> tensorI{1,C,H,W}

        // (5.2) Set image for detection
        RGB.copyTo(imageD);
        tensorD = transforms::apply(transformD, imageD);  // imageD{H_D,W_D,3} ==={ToTensor,etc.}===> tensorD{3,H_D,W_D}

        // (5.3) Inference and Detection
        tensorO = model->forward(tensorI);  // {1,C,H,W} ===> {S,{1,G,G,FF}}
        /*************************************************************************/
        tensorO_one = std::vector<torch::Tensor>(tensorO.size());
        for (size_t i = 0; i < tensorO.size(); i++) {
            tensorO_one.at(i) = tensorO.at(i)[0];
        }
        detect_result = detector(tensorO_one, { (float)tensorI.size(3), (float)tensorI.size(2) });  // tensorO_one{S,{G,G,FF}} ===> detect_result{ (ids{BB_n}, coords{BB_n,4}, probs{BB_n}) }
        /*************************************************************************/
        imageO = visualizer::draw_detections_des(tensorD.detach(), { std::get<0>(detect_result), std::get<1>(detect_result) }, std::get<2>(detect_result), class_names, label_palette, /*range=*/output_range);

        // (5.4) Calculate FPS
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 0.001;
        start = std::chrono::system_clock::now();
        switch (flag) {
        case true:
            flag = false;
            seconds_est = seconds;
            fps = 1.0 / seconds;
            break;
        default:
            seconds_est = (1.0 - alpha) * seconds_est + alpha * seconds;
            fps = (1.0 - alpha) * fps + alpha * (1.0 / seconds);
        }
        ss1.str(""); ss1.clear(std::stringstream::goodbit);
        ss2.str(""); ss2.clear(std::stringstream::goodbit);
        ss1 << std::setprecision(3) << fps;
        ss2 << std::setprecision(3) << seconds;
        std::cout << "FPS: " << ss1.str() << "[frame/second] (" << ss2.str() << "[second/frame])" << std::endl;

        // (5.5) Show the image in which the objects were detected
        cv::imshow("demo", imageO);
        key = cv::waitKey(1);
        if (key == 'q') break;

    }

    // Post Processing
    cap.release();
    cv::destroyAllWindows();

    // End Processing
    return;

}
