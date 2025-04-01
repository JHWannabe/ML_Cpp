#include "segmentation.h"


// -----------------------------------
// 0. Argument Function
// -----------------------------------
po::options_description parse_arguments(){

    po::options_description args("Options", 200, 30);
    
    args.add_options()

        // (1) Define for General Parameter
        ("help", "produce help message")
        ("dataset", po::value<std::string>(), "dataset name")
        ("size", po::value<size_t>()->default_value(256), "image width and height (x>=64)")
        ("nc", po::value<size_t>()->default_value(3), "input image channel : RGB=3, grayscale=1")
        ("nz", po::value<size_t>()->default_value(512), "dimensions of latent space")
        ("class_num", po::value<size_t>()->default_value(256), "total classes")
        ("gpu_id", po::value<int>()->default_value(0), "cuda device : 'x=-1' is cpu device")
        ("seed_random", po::value<bool>()->default_value(false), "whether to make the seed of random number in a random")
        ("seed", po::value<int>()->default_value(0), "seed of random number")

        // (2) Define for Training
        ("train", po::value<bool>()->default_value(false), "training mode on/off")
        ("train_in_dir", po::value<std::string>()->default_value("trainI"), "training input image directory : ./datasets/<dataset>/<train_in_dir>/<image files>")
        ("train_out_dir", po::value<std::string>()->default_value("trainO"), "training output image directory : ./datasets/<dataset>/<train_out_dir>/<image files>")
        ("epochs", po::value<size_t>()->default_value(200), "training total epoch")
        ("batch_size", po::value<size_t>()->default_value(32), "training batch size")
        ("train_load_epoch", po::value<std::string>()->default_value(""), "epoch of model to resume learning")
        ("save_epoch", po::value<size_t>()->default_value(20), "frequency of epoch to save model and optimizer")

        // (3) Define for Validation
        ("valid", po::value<bool>()->default_value(false), "validation mode on/off")
        ("valid_in_dir", po::value<std::string>()->default_value("validI"), "validation input image directory : ./datasets/<dataset>/<valid_in_dir>/<image files>")
        ("valid_out_dir", po::value<std::string>()->default_value("validO"), "validation output image directory : ./datasets/<dataset>/<valid_out_dir>/<image files>")
        ("valid_batch_size", po::value<size_t>()->default_value(1), "validation batch size")
        ("valid_freq", po::value<size_t>()->default_value(1), "validation frequency to training epoch")

        // (4) Define for Test
        ("test", po::value<bool>()->default_value(false), "test mode on/off")
        ("test_in_dir", po::value<std::string>()->default_value("testI"), "test input image directory : ./datasets/<dataset>/<test_in_dir>/<image files>")
        ("test_out_dir", po::value<std::string>()->default_value("testO"), "test output image directory : ./datasets/<dataset>/<test_out_dir>/<image files>")
        ("test_load_epoch", po::value<std::string>()->default_value("latest"), "training epoch used for testing")
        ("test_result_dir", po::value<std::string>()->default_value("test_result"), "test result directory : ./<test_result_dir>")

        // (5) Define for Network Parameter
        ("lr", po::value<float>()->default_value(1e-4), "learning rate")
        ("beta1", po::value<float>()->default_value(0.5), "beta 1 in Adam of optimizer method")
        ("beta2", po::value<float>()->default_value(0.999), "beta 2 in Adam of optimizer method")
        ("nf", po::value<size_t>()->default_value(64), "the number of filters in convolution layer closest to image")
        ("no_dropout", po::value<bool>()->default_value(false), "Dropout off/on")

    ;
    
    // End Processing
    return args;
}


// -----------------------------------
// 1. Main Function
// -----------------------------------
int mainSegmentation(int argc, const char *argv[], std::string file_path){
    if (!std::filesystem::exists(file_path))
    {
        return 1;
    }
    mINI::INIFile file(file_path);
    mINI::INIStructure ini;
    // now we can read the file
    if (!file.read(ini))
        return 1;

    // (1) Extract Arguments
    po::options_description args = parse_arguments();
    po::variables_map vm{};
    po::store(po::parse_command_line(argc, argv, args), vm);
    po::notify(vm);
    if (vm.empty() || vm.count("help")){
        std::cout << args << std::endl;
        return 1;
    }
    
    // (2) Select Device
    torch::Device device = Set_Device(ini);
    std::cout << "using device = " << device << std::endl;

    // (3) Set Seed
    if (stringToBool(ini["General"]["seed_random"]))
    {
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
    } torch::globalContext().setBenchmarkCuDNN(false);

    // (4) Set Transforms
    std::vector<transforms_Compose> transformI{
        transforms_Resize(cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"])), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if (std::stol(ini["General"]["input_channel"]) == 1) {
        transformI.insert(transformI.begin(), transforms_Grayscale(1));
    }
    std::vector<transforms_Compose> transformO{
        transforms_Resize(cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"])), cv::INTER_NEAREST),  // {IH,IW,1} ===method{OW,OH}===> {OH,OW,1}
        // transforms_ConvertIndex(255, 21),                                                                   // pixel_value=255 ===> pixel_value=21
        //transforms_ToTensorLabel()                                                                         // Mat Image ===> Tensor Label
        transforms_ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    
    // (5) Define Network
    UNet unet(ini);
    unet->to(device);
    
    // (6) Make Directories
    std::string dir = "./checkpoints/" + ini["General"]["dataset"];
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(ini, unet, "UNet");

    // (8.1) Training Phase
    if (stringToBool(ini["Training"]["train"])) {
        Set_Options(ini, argc, argv, args, "train");
        train(ini, device, unet, transformI, transformO);
    }

    // (8.2) Test Phase
    if (stringToBool(ini["Test"]["test"])) {
        Set_Options(ini, argc, argv, args, "test");
        test(ini, device, unet, transformI, transformO);
    }

    // End Processing
    return 0;

}


// -----------------------------------
// 2. Device Setting Function
// -----------------------------------
torch::Device Set_Device(mINI::INIStructure& ini)
{
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

// -----------------------------------
// 3. Model Parameters Setting Function
// -----------------------------------
void Set_Model_Params(mINI::INIStructure& ini, UNet& model, const std::string name) {

    // (1) Make Directory
    std::string dir = "checkpoints/" + ini["General"]["dataset"] + "/model_params/";
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

// -----------------------------------
// 4. Options Setting Function
// -----------------------------------
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], po::options_description& args, const std::string mode) {

    // (1) Make Directory
    std::string dir = "checkpoints/" + ini["General"]["dataset"] + "/options/";
    fs::create_directories(dir);

    // (2) Terminal Output
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << args << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

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
    ofs << "--------------------------------------------" << std::endl;
    ofs << args << std::endl;
    ofs << "--------------------------------------------" << std::endl << std::endl;

    // (3.3) File Close
    ofs.close();

    // End Processing
    return;

}

// -----------------------------------
// 5. change string to lower
// -----------------------------------
bool stringToBool(const std::string& str)
{
    // 소문자로 변환하여 비교하기 위해 문자열 복사본 생성
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    // 다양한 "true"에 해당하는 값을 true로 변환
    if (lowerStr == "true" || lowerStr == "1") {
        return true;
    }
    else if (lowerStr == "false" || lowerStr == "0") {
        return false;
    }

    // 예외 처리, 다른 값일 경우 false 또는 예외 발생
    throw std::invalid_argument("Invalid boolean string value");
}

void test(mINI::INIStructure& ini, torch::Device& device, UNet& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformO) {

    // (0) Initialization and Declaration
    size_t correct, correct_per_class, total_class_pixel, class_count;
    float ave_loss;
    double seconds, ave_time;
    double pixel_wise_accuracy, ave_pixel_wise_accuracy;
    double mean_accuracy, ave_mean_accuracy;
    std::string path, result_dir, fname;
    std::string input_dir, output_dir;
    std::ofstream ofs;
    std::chrono::system_clock::time_point start, end;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> data;
    torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
    torch::Tensor loss;
    datasets::ImageFolderSegmentWithPaths dataset;
    DataLoader::ImageFolderSegmentWithPaths dataloader;

    // (1) Get Test Dataset
    input_dir = "../Segmentation/datasets/" + ini["General"]["dataset"] + '/' + ini["Test"]["test_in_dir"];
    output_dir = "../Segmentation/datasets/" + ini["General"]["dataset"] + '/' + ini["Test"]["test_out_dir"];

    dataset = datasets::ImageFolderSegmentWithPaths(input_dir, output_dir, transformI, transformO);
    dataloader = DataLoader::ImageFolderSegmentWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
    std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

    // (2) Get Model
    path = "../Segmentation/checkpoints/" + ini["General"]["dataset"] + "/models/epoch_" + ini["Test"]["test_load_epoch"] + ".pth";
    torch::load(model, path, device);

    // (3) Set Loss Function
    auto criterion = CEDiceLoss();

    // (4) Initialization of Value
    ave_loss = 0.0;
    ave_pixel_wise_accuracy = 0.0;
    ave_mean_accuracy = 0.0;
    ave_time = 0.0;

    // (5) Tensor Forward
    torch::NoGradGuard no_grad;
    model->eval();
    result_dir = ini["Test"]["test_result_dir"];
    fs::create_directories(result_dir);
    ofs.open(result_dir + "/loss.txt", std::ios::out);
    while (dataloader(data)) {

        image = std::get<0>(data).to(device);
        label = std::get<1>(data).to(device);

        if (!device.is_cpu()) torch::cuda::synchronize();
        start = std::chrono::system_clock::now();

        output = model->forward(image);

        if (!device.is_cpu()) torch::cuda::synchronize();
        end = std::chrono::system_clock::now();
        seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

        loss = criterion(output, label);

        output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
        correct = (label == output_argmax).sum().item<int64_t>();
        pixel_wise_accuracy = (double)correct / (double)(label.size(0) * label.size(1) * label.size(2));

        class_count = 0;
        mean_accuracy = 0.0;
        for (size_t i = 0; i < std::get<4>(data).size(); i++) {
            answer_mask = torch::full({ label.size(0), label.size(1), label.size(2) }, /*value=*/(int64_t)i, torch::TensorOptions().dtype(torch::kLong)).to(device);
            total_class_pixel = (label == answer_mask).sum().item<int64_t>();
            if (total_class_pixel != 0) {
                response_mask = torch::full({ label.size(0), label.size(1), label.size(2) }, /*value=*/2, torch::TensorOptions().dtype(torch::kLong)).to(device);
                correct_per_class = (((label == output_argmax).to(torch::kLong) + (label == answer_mask).to(torch::kLong)) == response_mask).sum().item<int64_t>();
                mean_accuracy += (double)correct_per_class / (double)total_class_pixel;
                class_count++;
            }
        }
        mean_accuracy = mean_accuracy / (double)class_count;

        ave_loss += loss.item<float>();
        ave_pixel_wise_accuracy += pixel_wise_accuracy;
        ave_mean_accuracy += mean_accuracy;
        ave_time += seconds;

        std::cout << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " pixel-wise-accuracy:" << pixel_wise_accuracy << " mean-accuracy:" << mean_accuracy << std::endl;
        ofs << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " pixel-wise-accuracy:" << pixel_wise_accuracy << " mean-accuracy:" << mean_accuracy << std::endl;

        fname = result_dir + '/' + std::get<3>(data).at(0);
        visualizer::save_label(output_argmax.detach(), fname, std::get<4>(data), /*cols=*/1, /*padding=*/0);

    }

    // (6) Calculate Average
    ave_loss = ave_loss / (float)dataset.size();
    ave_pixel_wise_accuracy = ave_pixel_wise_accuracy / (double)dataset.size();
    ave_mean_accuracy = ave_mean_accuracy / (double)dataset.size();
    ave_time = ave_time / (double)dataset.size();

    // (7) Average Output
    std::cout << "<All> cross-entropy:" << ave_loss << " pixel-wise-accuracy:" << ave_pixel_wise_accuracy << " mean-accuracy:" << ave_mean_accuracy << " (time:" << ave_time << ')' << std::endl;
    ofs << "<All> cross-entropy:" << ave_loss << " pixel-wise-accuracy:" << ave_pixel_wise_accuracy << " mean-accuracy:" << ave_mean_accuracy << " (time:" << ave_time << ')' << std::endl;

    // Post Processing
    ofs.close();

    // End Processing
    return;

}

void train(mINI::INIStructure& ini, torch::Device& device, UNet& model, std::vector<transforms_Compose>& transformI, std::vector<transforms_Compose>& transformO) {

    constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
    constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset
    constexpr size_t save_sample_iter = 50;  // the frequency of iteration to save sample images
    constexpr std::string_view extension = "png";  // the extension of file name to save sample images

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t epoch, iter;
    size_t total_iter;
    size_t start_epoch, total_epoch;
    std::string date, date_out;
    std::string buff, latest;
    std::string checkpoint_dir, save_images_dir, path;
    std::string input_dir, output_dir;
    std::string valid_input_dir, valid_output_dir;
    std::stringstream ss;
    std::ifstream infoi;
    std::ofstream ofs, init, infoo;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> mini_batch;
    torch::Tensor loss, image, label, output, output_argmax;
    datasets::ImageFolderSegmentWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderSegmentWithPaths dataloader, valid_dataloader;
    visualizer::graph train_loss, valid_loss;
    progress::display* show_progress;
    progress::irregular irreg_progress;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    input_dir = "../Segmentation/datasets/" + ini["General"]["dataset"] + "/" + ini["Training"]["train_in_dir"];
    output_dir = "../Segmentation/datasets/" + ini["General"]["dataset"] + "/" + ini["Training"]["train_out_dir"];
    dataset = datasets::ImageFolderSegmentWithPaths(input_dir, output_dir, transformI, transformO);
    dataloader = DataLoader::ImageFolderSegmentWithPaths(dataset, std::stol(ini["Training"]["batch_size"]), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // (2) Get Validation Dataset
    if (stringToBool(ini["Validation"]["valid"])) {
        valid_input_dir = "../Segmentation/datasets/" + ini["General"]["dataset"] + "/" + ini["Validation"]["valid_in_dir"];
        valid_output_dir = "../Segmentation/datasets/" + ini["General"]["dataset"] + "/" + ini["Validation"]["valid_out_dir"];
        valid_dataset = datasets::ImageFolderSegmentWithPaths(valid_input_dir, valid_output_dir, transformI, transformO);
        valid_dataloader = DataLoader::ImageFolderSegmentWithPaths(valid_dataset, std::stol(ini["Validation"]["valid_batch_size"]), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(std::stof(ini["Network"]["lr"])).betas({ std::stof(ini["Network"]["beta1"]), std::stof(ini["Network"]["beta2"]) }));

    // (4) Set Loss Function
    auto criterion = CEDiceLoss();

    // (5) Make Directories
    checkpoint_dir = "../Segmentation/checkpoints/" + ini["General"]["dataset"];
    path = checkpoint_dir + "/models";  fs::create_directories(path);
    path = checkpoint_dir + "/optims";  fs::create_directories(path);
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    save_images_dir = checkpoint_dir + "/samples";  fs::create_directories(save_images_dir);

    // (6) Set Training Loss for Graph
    path = checkpoint_dir + "/graph";
    train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{ "Classification" });
    if (stringToBool(ini["Validation"]["valid"])) {
        valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{ "Classification" });
    }

    // (7) Get Weights and File Processing
    if (ini["Training"]["train_load_epoch"] == "") {
        model->apply(weights_init);
        ofs.open(checkpoint_dir + "/log/train.txt", std::ios::out);
        if (stringToBool(ini["Validation"]["valid"])) {
            init.open(checkpoint_dir + "/log/valid.txt", std::ios::trunc);
            init.close();
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

    // (2) Training per Epoch
    irreg_progress.restart(start_epoch - 1, total_epoch);
    for (epoch = start_epoch; epoch <= total_epoch; epoch++) {

        model->train();
        ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
        show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{ epoch, total_epoch }, /*loss_=*/{ "classify" });

        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)) {

            // -----------------------------------
            // c1. U-Net Training Phase
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
            show_progress->increment(/*loss_value=*/{ loss.item<float>() });
            ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
            ofs << "classify:" << loss.item<float>() << "(ave:" << show_progress->get_ave(0) << ')' << std::endl;

            // -----------------------------------
            // c3. Save Sample Images
            // -----------------------------------
            //iter = show_progress->get_iters();
            //if (iter % save_sample_iter == 1) {
            //    ss.str(""); ss.clear(std::stringstream::goodbit);
            //    ss << save_images_dir << "/epoch_" << epoch << "-iter_" << iter << '.' << extension;
            //    output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
            //    visualizer::save_label(output_argmax.detach(), ss.str(), std::get<4>(mini_batch));
            //}

        }

        // -----------------------------------
        // b2. Record Loss (epoch)
        // -----------------------------------
        train_loss.plot(/*base=*/epoch, /*value=*/show_progress->get_ave());

        // -----------------------------------
        // b3. Save Sample Images
        // -----------------------------------
        //ss.str(""); ss.clear(std::stringstream::goodbit);
        //ss << save_images_dir << "/epoch_" << epoch << "-iter_" << show_progress->get_iters() << '.' << extension;
        //output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
        //visualizer::save_label(output_argmax.detach(), ss.str(), std::get<4>(mini_batch));
        //delete show_progress;

        // -----------------------------------
        // b4. Validation Mode
        // -----------------------------------
        if (stringToBool(ini["Validation"]["valid"]) && ((epoch - 1) % std::stol(ini["Validation"]["valid_freq"]) == 0)) {
            valid(ini, valid_dataloader, device, criterion, model, epoch, valid_loss);
        }

        // -----------------------------------
        // b5. Save Model Weights and Optimizer Parameters
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
        // b6. Show Elapsed Time
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

void valid(mINI::INIStructure& ini, DataLoader::ImageFolderSegmentWithPaths& valid_dataloader, torch::Device& device, CEDiceLoss& criterion, UNet& model, const size_t epoch, visualizer::graph& writer) {

    // (0) Initialization and Declaration
    size_t correct, correct_per_class, total_class_pixel, class_count;
    size_t iteration;
    float ave_loss, total_loss;
    double pixel_wise_accuracy, ave_pixel_wise_accuracy, total_pixel_wise_accuracy;
    double mean_accuracy, ave_mean_accuracy, total_mean_accuracy;
    std::ofstream ofs;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>, std::vector<std::tuple<unsigned char, unsigned char, unsigned char>>> mini_batch;
    torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
    torch::Tensor loss;

    // (1) Tensor Forward per Mini Batch
    torch::NoGradGuard no_grad;
    model->eval();
    iteration = 0;
    total_loss = 0.0;
    total_pixel_wise_accuracy = 0.0;
    total_mean_accuracy = 0.0;
    while (valid_dataloader(mini_batch)) {

        image = std::get<0>(mini_batch).to(device);
        label = std::get<1>(mini_batch).to(device);

        output = model->forward(image);
        loss = criterion(output, label);

        output_argmax = output.exp().argmax(/*dim=*/1, /*keepdim=*/true);
        correct = (label == output_argmax).sum().item<int64_t>();
        pixel_wise_accuracy = (double)correct / (double)(label.size(0) * label.size(1) * label.size(2));

        class_count = 0;
        mean_accuracy = 0.0;
        for (size_t i = 0; i < std::get<4>(mini_batch).size(); i++) {
            answer_mask = torch::full({ label.size(0), label.size(1), label.size(2) }, /*value=*/(int64_t)i, torch::TensorOptions().dtype(torch::kLong)).to(device);
            total_class_pixel = (label == answer_mask).sum().item<int64_t>();
            if (total_class_pixel != 0) {
                response_mask = torch::full({ label.size(0), label.size(1), label.size(2) }, /*value=*/2, torch::TensorOptions().dtype(torch::kLong)).to(device);
                correct_per_class = (((label == output_argmax).to(torch::kLong) + (label == answer_mask).to(torch::kLong)) == response_mask).sum().item<int64_t>();
                mean_accuracy += (double)correct_per_class / (double)total_class_pixel;
                class_count++;
            }
        }
        mean_accuracy = mean_accuracy / (double)class_count;

        total_loss += loss.item<float>();
        total_pixel_wise_accuracy += pixel_wise_accuracy;
        total_mean_accuracy += mean_accuracy;
        iteration++;
    }

    // (2) Calculate Average Loss
    ave_loss = total_loss / (float)iteration;
    ave_pixel_wise_accuracy = total_pixel_wise_accuracy / (double)iteration;
    ave_mean_accuracy = total_mean_accuracy / (double)iteration;

    // (3.1) Record Loss (Log)
    ofs.open("../Segmentation/checkpoints/" + ini["General"]["dataset"] + "/log/valid.txt", std::ios::app);
    ofs << "epoch:" << epoch << '/' << std::stol(ini["Training"]["epochs"]) << ' ' << std::flush;
    ofs << "classify:" << ave_loss << ' ' << std::flush;
    ofs << "pixel-wise-accuracy:" << ave_pixel_wise_accuracy << ' ' << std::flush;
    ofs << "mean-accuracy:" << ave_mean_accuracy << std::endl;
    ofs.close();

    // (3.2) Record Loss (Graph)
    writer.plot(/*base=*/epoch, /*value=*/{ ave_loss });

    // End Processing
    return;

}