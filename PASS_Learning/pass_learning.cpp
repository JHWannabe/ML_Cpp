#include "pass_learning.h"

// -----------------------------------
// 0. Argument Function
// -----------------------------------
po::options_description parse_arguments() {

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
int mainPASSLearning(int argc, const char* argv[], std::string file_path) {
	if (!std::filesystem::exists(file_path)) return 1;

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
	if (vm.empty() || vm.count("help")) {
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
	std::vector<transforms_Compose> imageTransform = {
		transforms_ToTensor(),
		transforms_Normalize(
			std::vector<float>{0.485f, 0.456f, 0.406f},
			std::vector<float>{0.229f, 0.224f, 0.225f}
		)
	};

	std::vector<transforms_Compose> labelTransform = {
		transforms_ToTensor()
	};

	// (5) Define Network
	std::shared_ptr<Supervised> Model = std::make_shared<Supervised>(ini);
	Model->to(device);

	// (6) Make Directories
	std::string dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"];
	fs::create_directories(dir);

	// (7) Save Model Parameters
	Set_Model_Params(ini, Model, "Model");

	// (8.1) Training Phase
	if (stringToBool(ini["Training"]["train"])) {
		Set_Options(ini, argc, argv, args, "train");
		train(ini, device, Model, imageTransform, labelTransform);
	}

	// (8.2) Test Phase
	if (stringToBool(ini["Test"]["test"])) {
		Set_Options(ini, argc, argv, args, "test");
		test(ini, device, Model, imageTransform, labelTransform);
	}

	// End Processing
	return 0;

}

// -----------------------------------
// 2. Device Setting Function
// -----------------------------------
torch::Device Set_Device(mINI::INIStructure& ini)
{
	if (ini["General"]["gpu_id"] == "cpu") {
		// (2) CPU Type
		torch::Device device(torch::kCPU);
		return device;
	}
	else if (torch::cuda::is_available() && std::stoi(ini["General"]["gpu_id"]) >= 0) {
		// (1) GPU Type
		torch::Device device(torch::kCUDA, std::stoi(ini["General"]["gpu_id"]));
		return device;
	}

}

// -----------------------------------
// 3. Model Parameters Setting Function
// -----------------------------------
void Set_Model_Params(mINI::INIStructure& ini, std::shared_ptr<Supervised>& model, const std::string name) {

	// (1) Make Directory
	std::string dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["mode"] + "/model_params/";
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
	std::string dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["mode"] + "/options/";
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


// -----------------------------------
// 6. Confusion Matrix Function
// -----------------------------------
int count_images_in_label_folder(const std::string& base_path) {
	std::string label_path = base_path + "/label";
	if (!fs::exists(label_path) || !fs::is_directory(label_path)) return 0;

	int count = 0;
	for (const auto& entry : fs::directory_iterator(label_path)) {
		if (entry.is_regular_file()) {
			std::string ext = entry.path().extension().string();
			if (ext == ".jpg") {
				count++;
			}
		}
	}
	return count;
}

std::vector<std::vector<int>> compute_confusion_matrix_from_dirs(
	const std::string& goodPath,
	const std::string& ngPath,
	const std::string& overkillPath,
	const std::string& notfoundPath) {

	int TP = count_images_in_label_folder(ngPath);         // NG → NG
	int TN = count_images_in_label_folder(goodPath);       // GOOD → GOOD
	int FP = count_images_in_label_folder(overkillPath);   // GOOD → NG
	int FN = count_images_in_label_folder(notfoundPath);   // NG → GOOD

	// Confusion matrix: [[TP, FN], [FP, TN]]
	std::vector<std::vector<int>> cm = {
		{TP, FN},
		{FP, TN}
	};
	return cm;
}

Metrics compute_metrics_from_confusion_matrix(const std::vector<std::vector<int>>& cm) {
	int TP = cm[0][0];
	int FN = cm[0][1];
	int FP = cm[1][0];
	int TN = cm[1][1];

	double recall = 0.0, precision = 0.0, f1_score = 0.0;

	if (TP + FN > 0)
		recall = static_cast<double>(TP) / (TP + FN);

	if (TP + FP > 0)
		precision = static_cast<double>(TP) / (TP + FP);

	if (precision + recall > 0)
		f1_score = 2.0 * (precision * recall) / (precision + recall);

	return { recall, precision, f1_score, FN };
}


// -----------------------------------
// 7. Test Function
// -----------------------------------
void test(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform) {
	// (0) Initialization and Declaration
	torch::NoGradGuard no_grad;
	int yTrue;
	float ave_loss;
	double seconds, ave_time;
	double ave_pixel_wise_accuracy, pixel_accuracy;
	double ave_mean_accuracy;
	std::vector<torch::Tensor> yTrueList, probList;
	cv::Mat imgCv, labelCv;
	std::string path, result_dir, fname;
	std::string folderPath, goodPath, ngPath, overkillPath, notfoundPath;
	std::string input_dir, output_dir;
	std::ofstream ofs;
	std::chrono::system_clock::time_point start, end;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<int>, std::vector<cv::Mat>, std::vector<cv::Mat>> data;
	torch::Tensor input, label, output, output_argmax, answer_mask, response_mask;
	datasets::SegmentImageWithPaths dataset;
	DataLoader::SegmentImageWithPaths dataloader;

	// (1) Get Test Dataset
	input_dir = ini["Test"]["test_dir"];
	cv::Size resize = cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"]));

	dataset = datasets::SegmentImageWithPaths(input_dir, imageTransform, labelTransform, "test", resize);
	dataloader = DataLoader::SegmentImageWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
	std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

	// (2) Get Model
	path = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["mode"] + "/models/epoch_" + ini["Test"]["test_load_epoch"] + ".pth";
	torch::load(model, path, device);
	model->eval();

	// (3) Initialization of Value
	ave_loss = 0.0;
	ave_pixel_wise_accuracy = 0.0;
	ave_mean_accuracy = 0.0;
	ave_time = 0.0;
	pixel_accuracy = 0.0;

	// (4) Tensor Forward
	result_dir = ini["Test"]["test_result_dir"];
	fs::create_directories(result_dir);

	folderPath = result_dir + "/data";
	goodPath = folderPath + "/Good";
	ngPath = folderPath + "/NG";
	overkillPath = folderPath + "/Overkill";
	notfoundPath = folderPath + "/Notfound";

	for (const auto& path : { goodPath, ngPath, overkillPath, notfoundPath }) {
		fs::create_directories(path);
		fs::create_directories(path + "/label");
	}

	ofs.open(result_dir + "/loss.txt", std::ios::out);
	while (dataloader(data)) {
		input = std::get<0>(data).to(device);
		label = std::get<1>(data).to(device).squeeze();
		fname = std::get<2>(data).at(0);
		fname = fname.substr(0, fname.length() - 4);
		size_t last_slash = fname.rfind('/');
		if (last_slash != std::string::npos) {
			fname = fname.substr(last_slash + 1);
		}
		yTrue = std::get<3>(data).at(0);
		imgCv = std::get<4>(data).at(0);
		labelCv = std::get<5>(data).at(0);

		if (!device.is_cpu()) torch::cuda::synchronize();
		start = std::chrono::system_clock::now();

		output = model->forward(input);
		output = torch::sigmoid(output);

		if (!device.is_cpu()) torch::cuda::synchronize();
		end = std::chrono::system_clock::now();
		seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

		// 3. CPU로 이동 + detach
		torch::Tensor final_outputs = output.detach().to(torch::kCPU);
		torch::Tensor final_segmap = final_outputs.squeeze();  // torch::Tensor [H, W]

		// 5. threshold 적용
		float threshold = stod(ini["Test"]["threshold"]);
		torch::Tensor binary_segmap = (final_segmap > threshold).to(torch::kUInt8);  // [H, W], 값은 0 또는 1
		torch::Tensor combined_final_segmap = binary_segmap * 255;  // [H, W], 값은 0 또는 255

		//auto sub = final_segmap.squeeze(0).slice(0, 0, 5)   // H축 0~4
		//	.slice(1, 0, 5); // W축 0~4
		//std::cout << "Sub-tensor [H0-4, W0-4]:\n"
		//	<< sub << "\n";

		//sub = binary_segmap.squeeze(0).slice(0, 0, 5)   // H축 0~4
		//	.slice(1, 0, 5); // W축 0~4
		//std::cout << "Sub-tensor [H0-4, W0-4]:\n"
		//	<< sub << "\n";

		// 7. OpenCV로 변환
		int height = combined_final_segmap.size(0);
		int width = combined_final_segmap.size(1);
		combined_final_segmap = combined_final_segmap.contiguous();  // data_ptr 접근용
		cv::Mat segmapCv(height, width, CV_8UC1, combined_final_segmap.data_ptr());

		// 2. flatten label and prediction for decision
		bool is_all_zero = torch::all(combined_final_segmap == 0).item<bool>();
		int predicted_label = is_all_zero ? 1 : 0; // 1 = GOOD, 0 = NG

		// 3. 저장 디렉토리 결정
		std::string save_dir;
		if (yTrue == 1 && predicted_label == 1) save_dir = goodPath;
		else if (yTrue == 0 && predicted_label == 1) save_dir = notfoundPath;
		else if (yTrue == 0 && predicted_label == 0) save_dir = ngPath;
		else if (yTrue == 1 && predicted_label == 0) save_dir = overkillPath;

		// 4. Segmentation Map
		// TODO

		// 5. 저장
		cv::imwrite(save_dir + "/" + fname + ".jpg", imgCv);
		cv::imwrite(save_dir + "/" + fname + "_segmap.jpg", segmapCv);
		cv::imwrite(save_dir + "/label/" + fname + "_label.jpg", labelCv);

		//// 둘 다 CV_8UC1인지 확인
		//CV_Assert(segmapCv.type() == CV_8UC1 && labelCv.type() == CV_8UC1);
		//CV_Assert(segmapCv.size() == labelCv.size());

		//// 정확히 일치하는 픽셀 수 계산
		//cv::Mat match_mask;
		//cv::compare(segmapCv, labelCv, match_mask, cv::CMP_EQ); // 같으면 255, 다르면 0
		//int matched_pixels = cv::countNonZero(match_mask);

		//// 전체 픽셀 수
		//int total_pixels = segmapCv.rows * segmapCv.cols;

		//// 정확도 계산
		//pixel_accuracy = static_cast<double>(matched_pixels) / total_pixels;

		//ave_pixel_wise_accuracy += pixel_accuracy;
		//ave_time += seconds;

		//std::cout << '<' << std::get<2>(data).at(0) << "> pixel-wise-accuracy: " << pixel_accuracy << std::endl;
		//ofs << '<' << std::get<2>(data).at(0) << "> pixel-wise-accuracy: " << pixel_accuracy << std::endl;
	}

	//// (6) Calculate Average
	//ave_pixel_wise_accuracy = ave_pixel_wise_accuracy / (double)dataset.size();
	//ave_time = ave_time / (double)dataset.size();

	//// (7) Average Output
	//std::cout << "<All> pixel-wise-accuracy: " << ave_pixel_wise_accuracy << " (time:" << ave_time << ')' << std::endl;
	//ofs << "<All> pixel-wise-accuracy: " << ave_pixel_wise_accuracy << " (time:" << ave_time << ')' << std::endl;

	// (8) Calculate Confusion Matrix
	auto cm = compute_confusion_matrix_from_dirs(goodPath, ngPath, overkillPath, notfoundPath);
	Metrics m = compute_metrics_from_confusion_matrix(cm);

	// (9) Print Confusion Matrix
	std::cout << "Notfound: " << m.FN << ", Recall: " << m.recall << ", F1-score: " << m.f1_score << std::endl;
	ofs << "<All> Notfound: " << m.FN << ", Recall: " << m.recall << ", F1-score: " << m.f1_score << std::endl;

	// Post Processing
	ofs.close();

	// End Processing
	return;
}

// -----------------------------------
// 8. Train Function
// -----------------------------------
void train(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform) {

	constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
	constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 4;  // the number of workers to retrieve data from the validation dataset

	// -----------------------------------
	// a0. Initialization and Declaration
	// -----------------------------------

	size_t epoch, total_iter, start_epoch, total_epoch;
	std::string mode, date, date_out, buff, latest;
	std::string checkpoint_dir, path, input_dir, valid_input_dir;
	std::stringstream ss;
	std::ifstream infoi;
	std::ofstream ofs, init, infoo;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<int>, std::vector<cv::Mat>, std::vector<cv::Mat>> mini_batch;
	torch::Tensor loss, image, label, output;
	datasets::SegmentImageWithPaths dataset, valid_dataset;
	DataLoader::SegmentImageWithPaths dataloader, valid_dataloader;
	visualizer::graph train_loss, valid_loss;
	progress::display* show_progress;
	progress::irregular irreg_progress;


	// -----------------------------------
	// a1. Preparation
	// -----------------------------------

	// (0) Mode Setting
	mode = ini["Training"]["mode"];
	std::cout << "train mode : " << mode << " [super|unsuper]" << std::endl;
	if (mode == "super") {
		input_dir = ini["Training"]["train_super_dir"];
	}
	else if (mode == "unsuper") {
		input_dir = ini["Training"]["train_unsuper_dir"];
	}

	// (1) Get Training Dataset
	cv::Size resize = cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"]));
	dataset = datasets::SegmentImageWithPaths(input_dir, imageTransform, labelTransform, mode, resize);
	dataloader = DataLoader::SegmentImageWithPaths(dataset, std::stol(ini["Training"]["batch_size"]), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
	std::cout << "total training images : " << dataset.size() << std::endl;

	// (2) Get Validation Dataset
	if (stringToBool(ini["Validation"]["valid"])) {
		valid_input_dir = ini["Validation"]["valid_dir"];
		valid_dataset = datasets::SegmentImageWithPaths(valid_input_dir, imageTransform, labelTransform, mode, resize);
		valid_dataloader = DataLoader::SegmentImageWithPaths(valid_dataset, std::stol(ini["Validation"]["valid_batch_size"]), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
		std::cout << "total validation images : " << valid_dataset.size() << std::endl;
	}

	// (3) Set Optimizer Method and Scheduler
	auto optimizer = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(std::stof(ini["Network"]["lr"])).betas({ std::stof(ini["Network"]["beta1"]), std::stof(ini["Network"]["beta2"]) }));

	CosineAnnealingWarmupRestarts scheduler = CosineAnnealingWarmupRestarts(
		optimizer,
		std::stoi(ini["Training"]["epochs"]),
		std::stof(ini["Network"]["lr"]),
		std::stof(ini["Network"]["min_lr"]),
		int(std::stoi(ini["Training"]["epochs"]) * std::stof(ini["Network"]["warmup_ratio"]))
	);

	// (4) Set Loss Function
	auto criterion = Loss();

	// (5) Make Directories
	checkpoint_dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + mode;
	path = checkpoint_dir + "/models";  fs::create_directories(path);
	path = checkpoint_dir + "/optims";  fs::create_directories(path);
	path = checkpoint_dir + "/log";  fs::create_directories(path);

	// (6) Set Training Loss for Graph
	path = checkpoint_dir + "/graph";
	train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{ "PASS Learning" });
	if (stringToBool(ini["Validation"]["valid"])) {
		valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{ "PASS Learning" });
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
		std::cout << "Successfully loaded model from: " << path << std::endl;
		//path = checkpoint_dir + "/optims/epoch_" + ini["Training"]["train_load_epoch"] + ".pth";  torch::load(optimizer, path, device);
		//std::cout << "Successfully loaded optimizer from: " << path << std::endl;

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
		std::cout << std::endl;

		int count = 1;
		optimizer.zero_grad();

		// -----------------------------------
		// b1. Mini Batch Learning
		// -----------------------------------
		while (dataloader(mini_batch)) {

			// -----------------------------------
			// c1. U-Net Training Phase
			// -----------------------------------
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			cv::Mat imageCv = std::get<4>(mini_batch).at(0);
			cv::Mat labelCv = std::get<5>(mini_batch).at(0);
			output = model->forward(image).to(device);
			output = torch::sigmoid(output);

			//auto sub = output.squeeze(0).slice(0, 0, 5)   // H축 0~4
			//	.slice(1, 0, 5); // W축 0~4
			//std::cout << "Sub-tensor [H0-4, W0-4]:\n"
			//	<< sub << "\n";

			loss = criterion(output, label, stof(ini["Training"]["l1_weight"]), stof(ini["Training"]["focal_weight"]));

			if (mode == "unsuper") {
				py::gil_scoped_release no_gil;
				loss.backward();
				optimizer.step();
			}
			else if (mode == "super") {
				loss.backward();
				optimizer.step();
			}

			optimizer.zero_grad();
			scheduler.step(epoch);

			// 3. CPU로 이동 + detach
			torch::Tensor final_outputs = output.detach().to(torch::kCPU);
			torch::Tensor final_segmap = final_outputs.squeeze();  // torch::Tensor [H, W]

			// 5. threshold 적용
			float threshold = stod(ini["Test"]["threshold"]);
			torch::Tensor binary_segmap = (final_segmap > threshold).to(torch::kUInt8);  // [H, W], 값은 0 또는 1
			torch::Tensor combined_final_segmap = binary_segmap * 255;  // [H, W], 값은 0 또는 255

			//auto sub = final_segmap.squeeze(0).slice(0, 0, 5)   // H축 0~4
			//	.slice(1, 0, 5); // W축 0~4
			//std::cout << "Sub-tensor [H0-4, W0-4]:\n"
			//	<< sub << "\n";

			//sub = binary_segmap.squeeze(0).slice(0, 0, 5)   // H축 0~4
			//	.slice(1, 0, 5); // W축 0~4
			//std::cout << "Sub-tensor [H0-4, W0-4]:\n"
			//	<< sub << "\n";

			// 7. OpenCV로 변환
			int height = combined_final_segmap.size(0);
			int width = combined_final_segmap.size(1);
			combined_final_segmap = combined_final_segmap.contiguous();  // data_ptr 접근용
			cv::Mat segmapCv(height, width, CV_8UC1, combined_final_segmap.data_ptr());

			// 5. 저장
			//cv::imwrite("segmap.jpg", segmapCv);
			//cv::imwrite("seg_img.jpg", imageCv);
			//cv::imwrite("seg_label.jpg", labelCv*255);

			// -----------------------------------
			// c2. Record Loss (iteration)
			// -----------------------------------
			show_progress->increment(/*loss_value=*/{ loss.item<float>() });
			ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
			ofs << "classify:" << loss.item<float>() << "(ave:" << show_progress->get_ave(0) << ')' << std::endl;

			if (mode == "unsuper") {
				if (count % std::stoi(ini["Training"]["unsuper_count"]) == 0) {
					if (stringToBool(ini["Validation"]["valid"])) {
						valid(ini, valid_dataloader, device, criterion, model, epoch, valid_loss);
					}
					path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + "_" + std::to_string(count) + ".pth";  torch::save(model, path);
					path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + "_" + std::to_string(count) + ".pth";  torch::save(optimizer, path);
					path = checkpoint_dir + "/models/epoch_latest.pth";  torch::save(model, path);
					path = checkpoint_dir + "/optims/epoch_latest.pth";  torch::save(optimizer, path);
					infoo.open(checkpoint_dir + "/models/info.txt", std::ios::out);
					infoo << "latest = " << epoch << std::endl;
					infoo.close();
				}
			}
			count++;
		}

		std::cout << "Epoch [" << epoch << "] Learning Rate: "
			<< optimizer.param_groups()[0].options().get_lr() << std::endl;

		// -----------------------------------
		// b2. Record Loss (epoch)
		// -----------------------------------
		train_loss.plot(/*base=*/epoch, /*value=*/show_progress->get_ave());


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

// -----------------------------------
// 9. Valid Function
// -----------------------------------
void valid(mINI::INIStructure& ini, DataLoader::SegmentImageWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, std::shared_ptr<Supervised>& model, const size_t epoch, visualizer::graph& writer) {

	// (0) Initialization and Declaration
	size_t correct, class_count;
	size_t iteration;
	float ave_loss, total_loss;
	double pixel_wise_accuracy, ave_pixel_wise_accuracy, total_pixel_wise_accuracy;
	double mean_accuracy, ave_mean_accuracy, total_mean_accuracy;
	std::ofstream ofs;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<int>, std::vector<cv::Mat>, std::vector<cv::Mat>> mini_batch;
	torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
	torch::Tensor loss;
	float dice_weight = 0.6;
	float focal_weight = 0.4;

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
		//for (size_t i = 0; i < std::get<4>(mini_batch).size(); i++) {
		//	answer_mask = torch::full({ label.size(0), label.size(1), label.size(2) }, /*value=*/(int64_t)i, torch::TensorOptions().dtype(torch::kLong)).to(device);
		//	total_class_pixel = (label == answer_mask).sum().item<int64_t>();
		//	if (total_class_pixel != 0) {
		//		response_mask = torch::full({ label.size(0), label.size(1), label.size(2) }, /*value=*/2, torch::TensorOptions().dtype(torch::kLong)).to(device);
		//		correct_per_class = (((label == output_argmax).to(torch::kLong) + (label == answer_mask).to(torch::kLong)) == response_mask).sum().item<int64_t>();
		//		mean_accuracy += (double)correct_per_class / (double)total_class_pixel;
		//		class_count++;
		//	}
		//}
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
	ofs.open("D:/Cpp/DLL_Porting/PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/log/valid.txt", std::ios::app);
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
