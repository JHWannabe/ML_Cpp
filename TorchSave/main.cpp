//#include "classification.h"
//
//int main() {
//	std::string file_path = "../config/classification.ini";
//	int argc = 1;
//	const char* argv[] = { "Classification" };
//	return mainClassification(argc, argv, file_path);
//}

//#include "object_detection.h"
//
//int main() {
//	std::string file_path = "../config/object_detection.ini";
//	int argc = 1;
//	const char* argv[] = { "Object_Detection" };
//	return mainObjectDetection(argc, argv, file_path);
//}

#include "pass_learning.h"

int main() {
	std::string file_path = "../config/piston.ini";
	int argc = 1;
	const char* argv[] = { "PASS Learning" };
	return mainPASSLearning(argc, argv, file_path);
}

//#include "segmentation.h"
//
//int main() {
//	std::string file_path = "../config/segmentation.ini";
//	int argc = 1;
//	const char* argv[] = { "Segmentation" };
//	return mainSegmentation(argc, argv, file_path);
//}