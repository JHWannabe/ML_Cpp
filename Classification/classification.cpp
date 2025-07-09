#include "classification.h"

int mainClassification(int argc, const char* argv[], std::string file_path) {

	// (0) Check File Path
	if (!std::filesystem::exists(file_path)) { return 1; }
	mINI::INIFile file(file_path);
	mINI::INIStructure ini;

	// now we can read the file
	if (!file.read(ini)) return 1;

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
	}

	// (5) Define Network
	MC_ResNet model(ini);
	model->to(device);

	// (6) Make Directories
	std::string dir = "../Classification/checkpoints/" + ini["General"]["dataset"];
	fs::create_directories(dir);

	// (7) Save Model Parameters
	Set_Model_Params(ini, model, "ResNet");

	// (8) Set Class Names
	std::vector<std::string> class_names = Set_Class_Names(ini["General"]["class_list"], std::stol(ini["General"]["class_num"]));

	// (9.1) Training Phase
	if (stringToBool(ini["Training"]["train"])) {
		Set_Options(ini, argc, argv, "train");
		train(ini, device, model, class_names);
	}

	// (9.2) Test Phase
	if (stringToBool(ini["Test"]["test"])) {
		Set_Options(ini, argc, argv, "test");
		test(ini, device, model, class_names);
	}

	// End Processing
	return 0;
}

// -----------------------------------
// 1. Device Setting Function
// -----------------------------------
torch::Device Set_Device(mINI::INIStructure& ini)
{
	// (1) GPU Type
	std::string& gpu_id_str = ini["General"]["gpu_id"];
	int gpu_id = std::stoi(gpu_id_str);
	if (torch::cuda::is_available() && gpu_id >= 0) {
		torch::Device device(torch::kCUDA, gpu_id);
		return device;
	}

	// (2) CPU Type
	torch::Device device(torch::kCPU);
	return device;
}

// -----------------------------------
// 2. Model Parameters Setting Function
// -----------------------------------
void Set_Model_Params(mINI::INIStructure& ini, MC_ResNet& model, const std::string name) {

	// (1) Make Directory
	std::string dir = "../Classification/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"] + "/model_params/";
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
// 3. Class Names Setting Function
// -----------------------------------
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

// -----------------------------------
// 4. Options Setting Function
// -----------------------------------
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], const std::string mode) {

	// (1) Make Directory
	std::string dir = "../Classification/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"] + "/options/";
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

// -----------------------------------
// 5. change string to lower
// -----------------------------------
bool stringToBool(const std::string& str)
{
	// Create a copy of the string to convert to lowercase for comparison
	std::string lowerStr = str;
	std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

	// Convert various representations of "true" to true
	if (lowerStr == "true" || lowerStr == "1") {
		return true;
	}
	else if (lowerStr == "false" || lowerStr == "0") {
		return false;
	}
}

void train(mINI::INIStructure& ini, torch::Device& device, MC_ResNet& model, const std::vector<std::string> class_names)
{
	const bool train_shuffle = stringToBool(ini["Training"]["train_shuffle"]);  // whether to shuffle the training dataset
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

	// (0) Transform
	std::vector<transforms_Compose> transform{
		transforms_Resize(cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"])), cv::INTER_LINEAR), // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
		//transforms_CenterCrop(cv::Size(224, 224)),
		transforms_ToTensor(),                                                                                  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
	};
	if (std::stol(ini["General"]["input_channel"]) == 1) {
		transform.insert(transform.begin(), transforms_Grayscale(1));
	}

	// (1) Get Training Dataset
	dataroot = ini["Training"]["train_dir"];
	dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
	dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, std::stol(ini["Training"]["batch_size"]), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
	std::cout << "total training images : " << dataset.size() << std::endl;

	// (2) Get Validation Dataset
	//if (stringToBool(ini["Validation"]["valid"])) {
	//    valid_dataroot = "./datasets/" + ini["General"]["dataset"] + "/" + ini["Validation"]["valid_dir"];
	//    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
	//    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, std::stol(ini["Validation"]["valid_batch_size"]), /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
	//    std::cout << "total validation images : " << valid_dataset.size() << std::endl;
	//}

	// (3) Set Optimizer Method
	auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(std::stof(ini["Network"]["lr"])).weight_decay(1e-5).betas({ std::stof(ini["Network"]["beta1"]), std::stof(ini["Network"]["beta2"]) }));

	//// CosineAnnealingLR setting
	//double T_max = 100; // The period (number of epochs) in which the learning rate changes in one cycle
	//double eta_min = 1e-6; // Minimum learning rate
	//double eta_max = 1e-3; // Maximum learning rate
	//CosineAnnealingLR scheduler(optimizer, T_max, eta_min, eta_max);


	// (4) Set Loss Function
	auto criterion = Loss();

	// (5) Make Directories
	checkpoint_dir = "../Classification/checkpoints/" + ini["General"]["dataset"];
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

			for (const auto& pair : traced_model.named_parameters()) {
				auto param_name = pair.name;
				auto param_tensor = pair.value;

				// Find the parameter corresponding to param_name in model's named_parameters()
				auto params = model->named_parameters();
				// Use std::find_if to find the item whose param_name matches
				auto it = std::find_if(params.begin(), params.end(),
					[&](const auto& p) { return p.key() == param_name; });

				if (it != params.end()) {
					// Compare parameter sizes
					if (it->value().sizes() != param_tensor.sizes()) {
						std::cerr << "Shape mismatch for " << param_name
							<< ": model size: " << it->value().sizes()
							<< ", traced size: " << param_tensor.sizes() << std::endl;
						continue;
					}

					// Copy parameter
					try {
						torch::NoGradGuard no_grad;  // Disable gradient tracking
						it->value().copy_(param_tensor.detach());  // Prevent in-place operation
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

	float initial_lr = std::stof(ini["Network"]["lr"]);  // Initial learning rate
	int step_size = 10;       // Learning rate decay period
	float gamma = 0.1;        // Learning rate decay ratio

	// (2) Training per Epoch
	irreg_progress.restart(start_epoch - 1, total_epoch);
	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {

		model->train();
		ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
		show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{ epoch, total_epoch }, /*loss_=*/{ "classify" });
		std::cout << std::endl;

		// -----------------------------------
		// Learning rate scheduler: update learning rate every epoch
		// -----------------------------------
		if (epoch % step_size == 0 && epoch > 0) {
			float new_lr = initial_lr * std::pow(gamma, epoch / step_size);
			for (auto& group : optimizer.param_groups()) {
				group.options().set_lr(new_lr);
			}
		}

		std::cout << "Epoch [" << epoch << "] Learning Rate: "
			<< optimizer.param_groups()[0].options().get_lr() << std::endl;

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
			try {
				loss = criterion(output, label);
			}
			catch (const c10::Error& e) {
				std::cerr << "LibTorch c10::Error: " << e.what() << std::endl;
			}
			catch (const std::exception& e) {
				std::cerr << "std::exception: " << e.what() << std::endl;
			}

			loss.backward();
			optimizer.step();
			optimizer.zero_grad();

			// -----------------------------------
			// c2. Record Loss (iteration)
			// -----------------------------------
			show_progress->increment(/*loss_value=*/{ loss.item().toFloat() });
			ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
			ofs << "classify:" << loss.item<float>() << "(ave:" << show_progress->get_ave(0) << ')' << std::endl;
		}

		// CosineAnnealingLR scheduler update
		// scheduler.step(epoch);


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

void valid(mINI::INIStructure& ini, DataLoader::ImageFolderClassesWithPaths& valid_dataloader, torch::Device& device, Loss& criterion, MC_ResNet& model, const std::vector<std::string> class_names, const size_t epoch, visualizer::graph& writer, visualizer::graph& writer_accuracy, visualizer::graph& writer_each_accuracy) {

	constexpr size_t class_num_thresh = 10;  // threshold for the number of classes for determining whether to display accuracy graph for each class

	// (0) Initialization and Declaration
	size_t iteration;
	size_t mini_batch_size;
	size_t class_num;
	size_t total_match, total_counter;
	long int response, answer;
	float total_accuracy;
	float ave_loss, total_loss;
	std::ofstream ofs;
	std::vector<size_t> class_match, class_counter;
	std::vector<float> class_accuracy;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
	torch::Tensor loss, image, label, output, responses;

	// (1) Memory Allocation
	class_num = class_names.size();
	class_match = std::vector<size_t>(class_num, 0);
	class_counter = std::vector<size_t>(class_num, 0);
	class_accuracy = std::vector<float>(class_num, 0.0);

	// (2) Tensor Forward per Mini Batch
	torch::NoGradGuard no_grad;
	model->eval();
	iteration = 0;
	total_loss = 0.0;
	total_match = 0; total_counter = 0;
	while (valid_dataloader(mini_batch)) {

		image = std::get<0>(mini_batch).to(device);
		label = std::get<1>(mini_batch).to(device);
		mini_batch_size = image.size(0);

		output = model->forward(image);
		loss = criterion(output, label);

		responses = output.exp().argmax(/*dim=*/1);
		for (size_t i = 0; i < mini_batch_size; i++) {
			//response = responses[i].item<long int>();
			//answer = label[i].item<long int>();
			response = responses[i].item().toLong();
			answer = label[i].item().toLong();
			class_counter[answer]++;
			total_counter++;
			if (response == answer) {
				class_match[answer]++;
				total_match++;
			}
		}

		// total_loss += loss.item<float>();
		total_loss += loss.item().toFloat();
		iteration++;
	}

	// (3) Calculate Average Loss
	ave_loss = total_loss / (float)iteration;

	// (4) Calculate Accuracy
	for (size_t i = 0; i < class_num; i++) {
		class_accuracy[i] = (float)class_match[i] / (float)class_counter[i];
	}
	total_accuracy = (float)total_match / (float)total_counter;

	// (5.1) Record Loss (Log/Loss)
	ofs.open("checkpoints/" + ini["General"]["dataset"] + "/log/valid.txt", std::ios::app);
	ofs << "epoch:" << epoch << '/' << std::stol(ini["Training"]["epochs"]) << ' ' << std::flush;
	ofs << "classify:" << ave_loss << ' ' << std::flush;
	ofs << "accuracy:" << total_accuracy << std::endl;
	ofs.close();

	// (5.2) Record Loss (Log/Accuracy)
	ofs.open("checkpoints/" + ini["General"]["dataset"] + "/log/valid.csv", std::ios::app);
	if (epoch == 1) {
		ofs << "epoch," << std::flush;
		ofs << "accuracy," << std::flush;
		for (size_t i = 0; i < class_num; i++) {
			ofs << i << "(" << class_names.at(i) << ")," << std::flush;
		}
		ofs << std::endl;
	}
	ofs << epoch << '/' << std::stol(ini["Training"]["epochs"]) << ',' << std::flush;
	ofs << total_accuracy << ',' << std::flush;
	for (size_t i = 0; i < class_num; i++) {
		ofs << class_accuracy[i] << ',' << std::flush;
	}
	ofs << std::endl;
	ofs.close();

	// (5.3) Record Loss (Graph)
	writer.plot(/*base=*/epoch, /*value=*/{ ave_loss });
	writer_accuracy.plot(/*base=*/epoch, /*value=*/{ total_accuracy });
	if (class_num <= class_num_thresh) {
		writer_each_accuracy.plot(/*base=*/epoch, /*value=*/class_accuracy);
	}

	// End Processing
	return;

}

void test(mINI::INIStructure& ini, torch::Device& device, MC_ResNet& model, const std::vector<std::string> class_names)
{
	// (0) Initialization and Declaration
	size_t class_num;
	size_t match, counter;
	long int response, answer;
	char judge;
	float accuracy;
	float ave_loss;
	double seconds, ave_time;
	std::string path, result_dir;
	std::string dataroot;
	std::ofstream ofs, ofs2;
	std::chrono::system_clock::time_point start, end;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> data;
	torch::Tensor image, label, output;
	torch::Tensor loss;
	datasets::ImageFolderClassesWithPaths dataset;
	DataLoader::ImageFolderClassesWithPaths dataloader;

	// (0) Transform
	std::vector<transforms_Compose> transform{
		transforms_Resize(cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"])), cv::INTER_LINEAR), // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
		//transforms_CenterCrop(cv::Size(224, 224)),
		transforms_ToTensor(),                                                                                  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
	};
	if (std::stol(ini["General"]["input_channel"]) == 1) {
		transform.insert(transform.begin(), transforms_Grayscale(1));
	}

	// (1) Get Test Dataset
	dataroot = "../Classification/datasets/" + ini["General"]["dataset"] + "/" + ini["Test"]["test_dir"];
	dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
	dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
	std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

	// (2) Get Model
	path = "../Classification/checkpoints/" + ini["General"]["dataset"] + "/models/epoch_" + ini["Test"]["test_load_epoch"] + ".pth";
	torch::load(model, path, device);

	// (3) Set Loss Function
	auto criterion = Loss();

	// (4) Initialization of Value
	ave_loss = 0.0;
	ave_time = 0.0;
	match = 0;
	counter = 0;
	class_num = class_names.size();

	// (5) File Pre-processing
	result_dir = ini["Test"]["test_result_dir"];  fs::create_directories(result_dir);
	ofs.open(result_dir + "/loss.txt", std::ios::out);
	ofs2.open(result_dir + "/likelihood.csv", std::ios::out);
	ofs2 << "file name," << std::flush;
	ofs2 << "judge," << std::flush;
	for (size_t i = 0; i < class_num; i++) {
		ofs2 << i << "(" << class_names.at(i) << ")," << std::flush;
	}
	ofs2 << std::endl;

	// (6) Tensor Forward
	torch::NoGradGuard no_grad;
	model->eval();
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

		ave_loss += loss.item<float>();
		ave_time += seconds;

		output = output.exp();
		//response = output.argmax(/*dim=*/1).item<long int>();
		//answer = label[0].item<long int>();
		response = output.argmax(/*dim=*/1).item().toLong();
		answer = label[0].item().toLong();
		counter += 1;
		judge = 'F';
		if (response == answer) {
			match += 1;
			judge = 'T';
		}

		std::cout << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " judge:" << judge << " response:" << response << '(' << class_names.at(response) << ") answer:" << answer << '(' << class_names.at(answer) << ')' << std::endl;
		ofs << '<' << std::get<2>(data).at(0) << "> cross-entropy:" << loss.item<float>() << " judge:" << judge << " response:" << response << '(' << class_names.at(response) << ") answer:" << answer << '(' << class_names.at(answer) << ')' << std::endl;
		ofs2 << std::get<2>(data).at(0) << ',' << std::flush;
		ofs2 << judge << ',' << std::flush;
		output = output[0];  // {1, CN} ===> {CN}
		for (size_t i = 0; i < class_num; i++) {
			ofs2 << output[i].item<float>() << ',' << std::flush;
		}
		ofs2 << std::endl;

	}

	// (7.1) Calculate Average
	ave_loss = ave_loss / (float)dataset.size();
	ave_time = ave_time / (double)dataset.size();

	// (7.2) Calculate Accuracy
	accuracy = (float)match / float(counter);

	// (8) Average Output
	std::cout << "<All> cross-entropy:" << ave_loss << " accuracy:" << accuracy << " (time:" << ave_time << ')' << std::endl;
	ofs << "<All> cross-entropy:" << ave_loss << " accuracy:" << accuracy << " (time:" << ave_time << ')' << std::endl;

	// Post Processing
	ofs.close();
	ofs2.close();

	// End Processing
	return;
}