#include "pass_learning.h"

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

	// (1) Select Device
	torch::Device device = Set_Device(ini);
	std::cout << "using device = " << device << std::endl;

	// (2) Set Seed
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

	// (3) Set Transforms
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

	// (4) Define Network
	std::shared_ptr<Supervised> Model = std::make_shared<Supervised>(ini);
	Model->to(device);

	// (5) Make Directories
	std::string dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"];
	fs::create_directories(dir);

	// (6) Save Model Parameters
	Set_Model_Params(ini, Model, "Model");

	// (7.1) Training Phase
	if (stringToBool(ini["Training"]["train"])) {
		Set_Options(ini, argc, argv, "train");
		train(ini, device, Model, imageTransform, labelTransform);
	}

	// (7.2) Test Phase
	if (stringToBool(ini["Test"]["test"])) {
		Set_Options(ini, argc, argv, "test");
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
	if (torch::cuda::is_available() && std::stoi(ini["General"]["gpu_id"]) >= 0) {
		// (1) GPU Type
		torch::Device device(torch::kCUDA, std::stoi(ini["General"]["gpu_id"]));
		return device;
	}
	else {
		// (2) CPU Type
		torch::Device device(torch::kCPU);
		return device;
	}
	
}

// -----------------------------------
// 3. Model Parameters Setting Function
// -----------------------------------
void Set_Model_Params(mINI::INIStructure& ini, std::shared_ptr<Supervised>& model, const std::string name) {

	// (1) Make Directory
	std::string dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"] + "/model_params/";
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
void Set_Options(mINI::INIStructure& ini, int argc, const char* argv[], const std::string mode) {

	// (1) Make Directory
	std::string dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"] + "/options/";
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

	// Exception handling: throw if the value is not recognized as boolean
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

Metrics compute_metrics_from_confusion_matrix(
	const std::string& goodPath,
	const std::string& ngPath,
	const std::string& overkillPath,
	const std::string& notfoundPath) {

	int TP = count_images_in_label_folder(ngPath);         // NG ¡æ NG
	int TN = count_images_in_label_folder(goodPath);       // GOOD ¡æ GOOD
	int FP = count_images_in_label_folder(overkillPath);   // GOOD ¡æ NG (Overkill)
	int FN = count_images_in_label_folder(notfoundPath);   // NG ¡æ GOOD (Notfound)

	// Confusion matrix: [[TP, FN], [FP, TN]]
	std::vector<std::vector<int>> cm = {{TP, FN},{FP, TN}};

	double recall = 0.0, precision = 0.0, f1_score = 0.0;

	if (TP + FN > 0)
		recall = static_cast<double>(TP) / (TP + FN);

	if (TP + FP > 0)
		precision = static_cast<double>(TP) / (TP + FP);

	if (precision + recall > 0)
		f1_score = 2.0 * (precision * recall) / (precision + recall);

	return { recall, precision, f1_score, FN, FP };
}

// -----------------------------------
// 7. Train Function
// -----------------------------------
void train(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform) {
	// Dataset and dataloader configs
	constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
	constexpr size_t train_workers = 4;  // the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 0;  // the number of workers to retrieve data from the validation dataset

	// -----------------------------------
	// [0] Variable Declarations
	// -----------------------------------
	size_t epoch, total_iter, start_epoch, total_epoch;
	std::string mode, date, date_out, buff, latest;
	std::string checkpoint_dir, pretrain_dir, path, input_dir, valid_input_dir;
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
	// [1] Training Configuration Setup
	// -----------------------------------

	// Load training mode and dataset path
	mode = ini["Training"]["mode"];
	std::cout << "train mode : " << mode << " [super|unsuper]" << std::endl;
	if (mode == "super") {
		input_dir = ini["Training"]["train_super_dir"];
	}
	else if (mode == "unsuper") {
		input_dir = ini["Training"]["train_unsuper_dir"];
	}

	// Load and preprocess training dataset
	cv::Size resize = cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"]));
	dataset = datasets::SegmentImageWithPaths(input_dir, imageTransform, labelTransform, mode, resize);
	dataloader = DataLoader::SegmentImageWithPaths(dataset, std::stol(ini["Training"]["batch_size"]), /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
	std::cout << "total training images : " << dataset.size() << std::endl;

	// -----------------------------------
	// [2] Optimizer, Scheduler, Loss
	// -----------------------------------

	// Setup AdamW optimizer
	auto optimizer = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(std::stof(ini["Network"]["lr"])).betas({ std::stof(ini["Network"]["beta1"]), std::stof(ini["Network"]["beta2"]) }));

	// CosineAnnealingWarmupRestarts scheduler
	CosineAnnealingWarmupRestarts scheduler = CosineAnnealingWarmupRestarts(
		optimizer,
		std::stoi(ini["Training"]["epochs"]),
		std::stof(ini["Network"]["lr"]),
		std::stof(ini["Network"]["min_lr"]),
		int(std::stoi(ini["Training"]["epochs"]) * std::stof(ini["Network"]["warmup_ratio"]))
	);

	// Initialize loss function
	auto criterion = Loss();

	// -----------------------------------
	// [3] Output Directories and Logging
	// -----------------------------------

	// Create checkpoint and log directories
	checkpoint_dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + mode;
	path = checkpoint_dir + "/models";  fs::create_directories(path);
	path = checkpoint_dir + "/optims";  fs::create_directories(path);
	path = checkpoint_dir + "/log";  fs::create_directories(path);

	// Initialize loss graphs
	path = checkpoint_dir + "/graph";
	train_loss = visualizer::graph(path, /*gname_=*/"train_loss", /*label_=*/{ "PASS Learning" });
	if (stringToBool(ini["Validation"]["valid"])) {
		valid_loss = visualizer::graph(path, /*gname_=*/"valid_loss", /*label_=*/{ "PASS Learning" });
	}

	// -----------------------------------
	// [4] Load Previous Checkpoint
	// -----------------------------------

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
		pretrain_dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["pretrain_mode"];
		// Load model and optimizer state
		path = pretrain_dir + "/models/epoch_" + ini["Training"]["train_load_epoch"] + ".pth";  torch::load(model, path, device);
		std::cout << "Successfully loaded model from: " << path << std::endl;
		path = pretrain_dir + "/optims/epoch_" + ini["Training"]["train_load_epoch"] + ".pth";  torch::load(optimizer, path, device);
		std::cout << "Successfully loaded optimizer from: " << path << std::endl;

		ofs.open(pretrain_dir + "/log/train.txt", std::ios::app);
		ofs << std::endl << std::endl;
		if (ini["Training"]["train_load_epoch"] == "latest") {
			infoi.open(pretrain_dir + "/models/info.txt", std::ios::in);
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

	// Display Date
	date = progress::current_date();
	date = progress::separator_center("Train Loss (" + date + ")");
	std::cout << std::endl << std::endl << date << std::endl;
	ofs << date << std::endl;

	// -----------------------------------
	// [5] Training Loop
	// -----------------------------------

	// Set Parameters
	start_epoch++;
	total_iter = dataloader.get_count_max();
	total_epoch = std::stol(ini["Training"]["epochs"]);

	// Training per Epoch
	irreg_progress.restart(start_epoch - 1, total_epoch);
	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		model->train();
		ofs << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;
		show_progress = new progress::display(/*count_max_=*/total_iter, /*epoch=*/{ epoch, total_epoch }, /*loss_=*/{ "loss", "lr" });
		std::cout << std::endl;

		int count = 1;
		optimizer.zero_grad();

		// -----------------------------------
		// [6] Iterate Mini-Batches
		// -----------------------------------
		while (dataloader(mini_batch)) {
			// U-Net Training Phase
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			cv::Mat imageCv = std::get<4>(mini_batch).at(0);
			cv::Mat labelCv = std::get<5>(mini_batch).at(0);

			// Unfold into patches
			int B = image.size(0), C = image.size(1), H = image.size(2), W = image.size(3);
			int patchSize = W / 8;
			int stride = patchSize / 2;

			torch::Tensor input_patches = image.unfold(2, patchSize, stride).unfold(3, patchSize, stride);
			torch::Tensor label_patches = label.unfold(1, patchSize, stride).unfold(2, patchSize, stride);

			input_patches = input_patches.permute({ 0, 2, 3, 1, 4, 5 }).contiguous();
			input_patches = input_patches.view({ -1, C, patchSize, patchSize });
			label_patches = label_patches.contiguous().view({ -1, patchSize, patchSize });

			std::vector<torch::Tensor> output_total;

			for (int i = 0; i < input_patches.size(0); i++) {
				torch::Tensor patch_input = input_patches[i].unsqueeze(0);
				torch::Tensor patch_label = label_patches[i].unsqueeze(0);

				// Forward pass and loss
				output = model->forward(patch_input).to(device);
				output = torch::sigmoid(output);
				output_total.push_back(output);

				loss = criterion(output, patch_label, stof(ini["Training"]["l1_weight"]), stof(ini["Training"]["focal_weight"]));

				// Backward and optimize
				if (mode == "unsuper") {
					py::gil_scoped_release no_gil;
					loss.backward();
				}
				else if (mode == "super") {
					loss.backward();
				}

				optimizer.step();
				optimizer.zero_grad();
				scheduler.step(epoch);
			}

			// Reconstruct image from patches
			torch::Tensor final_outputs = torch::cat(output_total, 0);
			torch::Tensor final_outputs_flat = final_outputs.reshape({ B, -1, patchSize * patchSize }).permute({ 0, 2, 1 });
			torch::Tensor reconstructed = torch::nn::functional::fold(
				final_outputs_flat,
				torch::nn::functional::FoldFuncOptions({ H, W }, { patchSize, patchSize }).stride({ stride, stride })
			);

			// Post-process output
			torch::Tensor final_segmap = reconstructed.detach().to(torch::kCPU).squeeze();// torch::Tensor [H, W]

			// threshold
			float threshold = std::stod(ini["Test"]["threshold"]);
			torch::Tensor binary_segmap = (final_segmap > threshold).to(torch::kUInt8) * 255;

			// Save segmentation map
			cv::Mat segmapCv(final_segmap.size(0), final_segmap.size(1), CV_8UC1, binary_segmap.data_ptr());

			cv::imwrite("segmap.jpg", segmapCv);
			cv::imwrite("seg_img.jpg", imageCv);
			cv::imwrite("seg_label.jpg", labelCv * 255);

			// Logging loss
			show_progress->increment({ static_cast<float>(loss.item<float>()), static_cast<float>(optimizer.param_groups()[0].options().get_lr()) });
			ofs << "iters:" << show_progress->get_iters() << '/' << total_iter << ' ' << std::flush;
			ofs << "loss:" << loss.item<float>() << "(ave:" << show_progress->get_ave(0) << ')' << std::endl;

			// Save model if unsupervised
			// Save model and optimizer every unsuper_count iterations in unsupervised mode
			if (mode == "unsuper") {
				if (count % std::stoi(ini["Training"]["unsuper_count"]) == 0) {
					// Save model + optimizer
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

		// Record Loss (epoch)
		train_loss.plot(/*base=*/epoch, /*value=*/show_progress->get_ave());

		// Save Model Weights and Optimizer Parameters
		// Save model and optimizer every save_epoch epochs
		if (epoch % std::stol(ini["Training"]["save_epoch"]) == 0) {
			path = checkpoint_dir + "/models/epoch_" + std::to_string(epoch) + ".pth";  torch::save(model, path);
			path = checkpoint_dir + "/optims/epoch_" + std::to_string(epoch) + ".pth";  torch::save(optimizer, path);
		}
		path = checkpoint_dir + "/models/epoch_latest.pth";  torch::save(model, path);
		path = checkpoint_dir + "/optims/epoch_latest.pth";  torch::save(optimizer, path);
		infoo.open(checkpoint_dir + "/models/info.txt", std::ios::out);
		infoo << "latest = " << epoch << std::endl;
		infoo.close();

		// Time estimate every 10 epochs
		if (epoch % 10 == 0) {
			ss.str(""); ss.clear(std::stringstream::goodbit);
			irreg_progress.nab(epoch);
			ss << "elapsed = " << irreg_progress.get_elap() << '(' << irreg_progress.get_sec_per() << "sec/epoch)   ";
			ss << "remaining = " << irreg_progress.get_rem() << "   ";
			ss << "now = " << irreg_progress.get_date() << "   ";
			ss << "finish = " << irreg_progress.get_date_fin();
			date_out = ss.str();

			std::cout << date_out << std::endl << progress::separator() << std::endl;
			ofs << date_out << std::endl << progress::separator() << std::endl;
		}

		// -----------------------------------
		// [7] Validation Mode
		// -----------------------------------
		// Run validation every valid_freq epochs if enabled
		if (stringToBool(ini["Validation"]["valid"]) && ((epoch - 1) % std::stol(ini["Validation"]["valid_freq"]) == 0)) {
			valid(ini, device, model, imageTransform, labelTransform);
		}
	}

	// -----------------------------------
	// [7] Finalize
	// -----------------------------------
	ofs.close();
}

// -----------------------------------
// 8. Valid Function
// -----------------------------------
void valid(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform) {
	// -----------------------------------
	// [0] Variable Initialization
	// -----------------------------------
	torch::NoGradGuard no_grad;
	int yTrue;
	float ave_loss;
	double seconds, ave_time;
	double ave_pixel_wise_accuracy, pixel_accuracy;
	double ave_mean_accuracy;
	std::vector<torch::Tensor> yTrueList, probList;
	cv::Mat imgCv, labelCv;
	std::string path, checkpoint_dir, fname;
	std::string folderPath, goodPath, ngPath, overkillPath, notfoundPath;
	std::string input_dir, output_dir;
	std::ofstream ofs;
	std::chrono::system_clock::time_point start, end;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<int>, std::vector<cv::Mat>, std::vector<cv::Mat>> data;
	torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
	datasets::SegmentImageWithPaths dataset;
	DataLoader::SegmentImageWithPaths dataloader;

	// -----------------------------------
	// [1] Load Validation Dataset
	// -----------------------------------
	input_dir = ini["Validation"]["valid_dir"];
	cv::Size resize = cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"]));
	dataset = datasets::SegmentImageWithPaths(input_dir, imageTransform, labelTransform, "valid", resize);
	dataloader = DataLoader::SegmentImageWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
	std::cout << "total valid images : " << dataset.size() << std::endl;
	model->eval();

	// -----------------------------------
	// [3] Initialize Metric Accumulators
	// -----------------------------------
	ave_loss = 0.0;
	ave_pixel_wise_accuracy = 0.0;
	ave_mean_accuracy = 0.0;
	ave_time = 0.0;
	pixel_accuracy = 0.0;

	checkpoint_dir = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["mode"];
	ofs.open(checkpoint_dir + "/log/valid.txt", std::ios::app);

	// -----------------------------------
	// [5] Run Inference for Each Validation Image
	// -----------------------------------
	while (dataloader(data)) {
		image = std::get<0>(data).to(device);
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

		// Timing: Start
		if (!device.is_cpu()) torch::cuda::synchronize();
		start = std::chrono::system_clock::now();

		// Unfold image into patches
		int patchSize = std::stoi(ini["General"]["patch_size"]);
		int stride = patchSize / 2;
		int B = image.size(0);
		int C = image.size(1);
		int H = image.size(2);
		int W = image.size(3);

		torch::Tensor input_patches = image.unfold(2, patchSize, stride).unfold(3, patchSize, stride);
		torch::Tensor label_patches = label.unfold(1, patchSize, stride).unfold(2, patchSize, stride);

		input_patches = input_patches.permute({ 0, 2, 3, 1, 4, 5 }).contiguous();
		input_patches = input_patches.view({ -1, C, patchSize, patchSize });
		label_patches = label_patches.contiguous().view({ -1, patchSize, patchSize });

		std::vector<torch::Tensor> output_total;

		// Forward each patch through model
		for (int i = 0; i < input_patches.size(0); i++) {
			torch::Tensor patch_input = input_patches[i].unsqueeze(0);

			output = model->forward(patch_input).to(device);
			output = torch::sigmoid(output);
			output_total.push_back(output);
		}

		// Timing: End
		if (!device.is_cpu()) torch::cuda::synchronize();
		end = std::chrono::system_clock::now();
		seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

		// Reconstruct full segmentation map from patch outputs
		torch::Tensor final_outputs = torch::cat(output_total, 0);
		torch::Tensor final_outputs_flat = final_outputs.reshape({ B, -1, patchSize * patchSize }).permute({ 0, 2, 1 });

		torch::Tensor reconstructed = torch::nn::functional::fold(
			final_outputs_flat,
			torch::nn::functional::FoldFuncOptions({ H, W }, { patchSize, patchSize }).stride({ stride, stride })
		);

		// Move to CPU and apply threshold
		final_outputs = reconstructed.detach().to(torch::kCPU);
		torch::Tensor final_segmap = final_outputs.squeeze();  // torch::Tensor [H, W]
		float threshold = std::stod(ini["Test"]["threshold"]);
		torch::Tensor combined_final_segmap = (final_segmap > threshold).to(torch::kUInt8) * 255;

		// Convert to OpenCV format for saving
		int height = combined_final_segmap.size(0);
		int width = combined_final_segmap.size(1);
		combined_final_segmap = combined_final_segmap.contiguous();  // for data_ptr access
		cv::Mat segmapCv(height, width, CV_8UC1, combined_final_segmap.data_ptr());

		// -----------------------------------
		// [6] Pixel-wise Accuracy Calculation
		// -----------------------------------
		CV_Assert(segmapCv.type() == CV_8UC1 && labelCv.type() == CV_8UC1);
		CV_Assert(segmapCv.size() == labelCv.size());

		cv::Mat match_mask;
		cv::compare(segmapCv, labelCv, match_mask, cv::CMP_EQ); // 255 if equal, 0 if not
		int matched_pixels = cv::countNonZero(match_mask);

		int total_pixels = segmapCv.rows * segmapCv.cols;
		pixel_accuracy = static_cast<double>(matched_pixels) / total_pixels * 100;

		ave_pixel_wise_accuracy += pixel_accuracy;
		ave_time += seconds;

		// Print and log individual accuracy
		ofs << '<' << std::get<2>(data).at(0) << "> pixel-wise-accuracy: " << pixel_accuracy << std::endl;
	}

	// -----------------------------------
	// [7] Average Metrics Summary
	// -----------------------------------
	ave_pixel_wise_accuracy = ave_pixel_wise_accuracy / (double)dataset.size();
	ave_time = ave_time / (double)dataset.size();

	std::cout << "<All> pixel-wise-accuracy: " << ave_pixel_wise_accuracy << " (time:" << ave_time << ')' << std::endl;
	ofs << "<All> pixel-wise-accuracy: " << ave_pixel_wise_accuracy << " (time:" << ave_time << ')' << std::endl;

	// -----------------------------------
	// [8] Finalization
	// -----------------------------------
	ofs.close();
}

// -----------------------------------
// 9. Test Function
// -----------------------------------
void test(mINI::INIStructure& ini, torch::Device& device, std::shared_ptr<Supervised>& model, std::vector<transforms_Compose>& imageTransform, std::vector<transforms_Compose>& labelTransform) {
	// -----------------------------------
	// [0] Variable Initialization
	// -----------------------------------
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
	torch::Tensor image, label, output, output_argmax, answer_mask, response_mask;
	datasets::SegmentImageWithPaths dataset;
	DataLoader::SegmentImageWithPaths dataloader;

	// -----------------------------------
	// [1] Load Test Dataset
	// -----------------------------------
	input_dir = ini["Test"]["test_dir"];
	cv::Size resize = cv::Size(std::stol(ini["General"]["size_w"]), std::stol(ini["General"]["size_h"]));

	dataset = datasets::SegmentImageWithPaths(input_dir, imageTransform, labelTransform, "test", resize);
	dataloader = DataLoader::SegmentImageWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
	std::cout << "total test images : " << dataset.size() << std::endl << std::endl;

	// -----------------------------------
	// [2] Load Trained Model
	// -----------------------------------
	path = "../PASS_Learning/checkpoints/" + ini["General"]["dataset"] + "/" + ini["Training"]["mode"] + "/models/epoch_" + ini["Test"]["test_load_epoch"] + ".pth";
	torch::load(model, path, device);
	model->eval();

	// -----------------------------------
	// [3] Initialize Metric Accumulators
	// -----------------------------------
	ave_loss = 0.0;
	ave_pixel_wise_accuracy = 0.0;
	ave_mean_accuracy = 0.0;
	ave_time = 0.0;
	pixel_accuracy = 0.0;

	// -----------------------------------
	// [4] Prepare Output Directories
	// -----------------------------------
	result_dir = ini["Test"]["test_result_dir"] + "/" + ini["General"]["dataset"];
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

	// -----------------------------------
	// [5] Run Inference for Each Test Image
	// -----------------------------------
	while (dataloader(data)) {
		image = std::get<0>(data).to(device);
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

		// Timing: Start
		if (!device.is_cpu()) torch::cuda::synchronize();
		start = std::chrono::system_clock::now();

		// Unfold image into patches
		int patchSize = std::stoi(ini["General"]["patch_size"]);
		int stride = patchSize / 2;
		int B = image.size(0);
		int C = image.size(1);
		int H = image.size(2);
		int W = image.size(3);

		torch::Tensor input_patches = image.unfold(2, patchSize, stride).unfold(3, patchSize, stride);
		torch::Tensor label_patches = label.unfold(1, patchSize, stride).unfold(2, patchSize, stride);

		input_patches = input_patches.permute({ 0, 2, 3, 1, 4, 5 }).contiguous();
		input_patches = input_patches.view({ -1, C, patchSize, patchSize });
		label_patches = label_patches.contiguous().view({ -1, patchSize, patchSize });

		std::vector<torch::Tensor> output_total;

		// Forward each patch through model
		for (int i = 0; i < input_patches.size(0); i++) {
			torch::Tensor patch_input = input_patches[i].unsqueeze(0);

			output = model->forward(patch_input).to(device);
			output = torch::sigmoid(output);
			output_total.push_back(output);
		}

		// Timing: End
		if (!device.is_cpu()) torch::cuda::synchronize();
		end = std::chrono::system_clock::now();
		seconds = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 * 0.001;

		// Reconstruct full segmentation map from patch outputs
		torch::Tensor final_outputs = torch::cat(output_total, 0);
		torch::Tensor final_outputs_flat = final_outputs.reshape({ B, -1, patchSize * patchSize }).permute({ 0, 2, 1 });

		torch::Tensor reconstructed = torch::nn::functional::fold(
			final_outputs_flat,
			torch::nn::functional::FoldFuncOptions({ H, W }, { patchSize, patchSize }).stride({ stride, stride })
		);

		// Move to CPU and apply threshold
		final_outputs = reconstructed.detach().to(torch::kCPU);
		torch::Tensor final_segmap = final_outputs.squeeze();  // torch::Tensor [H, W]
		float threshold = std::stod(ini["Test"]["threshold"]);
		torch::Tensor combined_final_segmap = (final_segmap > threshold).to(torch::kUInt8) * 255;

		// Convert to OpenCV format for saving
		int height = combined_final_segmap.size(0);
		int width = combined_final_segmap.size(1);
		combined_final_segmap = combined_final_segmap.contiguous();  // for data_ptr access
		cv::Mat segmapCv(height, width, CV_8UC1, combined_final_segmap.data_ptr());

		// -----------------------------------
		// [6] Classification from Segmentation
		// -----------------------------------
		bool is_all_zero = torch::all(combined_final_segmap == 0).item<bool>();
		int predicted_label = is_all_zero ? 1 : 0; // 1 = GOOD, 0 = NG

		// Determine folder to save based on GT and predicted class
		std::string save_dir;
		if (yTrue == 1 && predicted_label == 1) save_dir = goodPath;
		else if (yTrue == 0 && predicted_label == 1) save_dir = notfoundPath;
		else if (yTrue == 0 && predicted_label == 0) save_dir = ngPath;
		else if (yTrue == 1 && predicted_label == 0) save_dir = overkillPath;

		// Save raw image, segmentation map, label image
		cv::imwrite(save_dir + "/" + fname + ".jpg", imgCv);
		cv::imwrite(save_dir + "/" + fname + "_segmap.jpg", segmapCv);
		cv::imwrite(save_dir + "/label/" + fname + "_label.jpg", labelCv);

		// -----------------------------------
		// [7] Pixel-wise Accuracy Calculation
		// -----------------------------------
		CV_Assert(segmapCv.type() == CV_8UC1 && labelCv.type() == CV_8UC1);
		CV_Assert(segmapCv.size() == labelCv.size());

		cv::Mat match_mask;
		cv::compare(segmapCv, labelCv, match_mask, cv::CMP_EQ); // 255 if equal, 0 if not
		int matched_pixels = cv::countNonZero(match_mask);

		int total_pixels = segmapCv.rows * segmapCv.cols;
		pixel_accuracy = static_cast<double>(matched_pixels) / total_pixels * 100;

		ave_pixel_wise_accuracy += pixel_accuracy;
		ave_time += seconds;

		// Print and log individual accuracy
		std::cout << '<' << std::get<2>(data).at(0) << "> pixel-wise-accuracy: " << pixel_accuracy << std::endl;
		ofs << '<' << std::get<2>(data).at(0) << "> pixel-wise-accuracy: " << pixel_accuracy << std::endl;
	}

	// -----------------------------------
	// [8] Average Metrics Summary
	// -----------------------------------
	ave_pixel_wise_accuracy = ave_pixel_wise_accuracy / (double)dataset.size();
	ave_time = ave_time / (double)dataset.size();

	std::cout << "<All> pixel-wise-accuracy: " << ave_pixel_wise_accuracy << " (time:" << ave_time << ')' << std::endl;
	ofs << "<All> pixel-wise-accuracy: " << ave_pixel_wise_accuracy << " (time:" << ave_time << ')' << std::endl;

	// -----------------------------------
	// [9] Confusion Matrix and Metrics
	// -----------------------------------
	Metrics m = compute_metrics_from_confusion_matrix(goodPath, ngPath, overkillPath, notfoundPath);

	// Print evaluation summary
	std::cout << "Notfound: " << m.FN << ", Overkill: " << m.FP << ", Recall: " << m.recall << ", F1-score: " << m.f1_score << std::endl;
	ofs << "<All> Notfound: " << m.FN << ", Overkill: " << m.FP << ", Recall: " << m.recall << ", F1-score: " << m.f1_score << std::endl;

	// -----------------------------------
	// [10] Finalization
	// -----------------------------------
	ofs.close();
}
