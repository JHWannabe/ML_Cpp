#include "diffusion.h"

// Create the Python interpreter only once globally
static std::unique_ptr<py::scoped_interpreter> g_interpreter;

// Initialize the Python interpreter if it hasn't been initialized yet
void initializePython() {
    if (!g_interpreter) {
        g_interpreter = std::make_unique<py::scoped_interpreter>();
        // Optionally release the GIL here if needed
        // py::gil_scoped_release release;
    }
}

// Run stable diffusion on the given file path and return the result as a cv::Mat
cv::Mat stableDiffusion(std::string& file_path) {
    // Initialize the interpreter if it is not already running
    if (!g_interpreter)
        initializePython();
    
    // Acquire the Python GIL
    py::gil_scoped_acquire gil;

    // Add the script path to sys.path with highest priority
    auto sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "../PASS_Learning/");

    // Import the stableDiffusion module and call the anomaly_source function
    auto sd_module = py::module_::import("stableDiffusion");
    auto array = sd_module
        .attr("anomaly_source")(file_path)
        .cast<py::array_t<uint8_t>>();

    // Convert the returned numpy array to cv::Mat
    auto info = array.request();
    int h = info.shape[0], w = info.shape[1];
    int c = (info.ndim == 3 ? info.shape[2] : 1);
    cv::Mat mat(h, w, c == 1 ? CV_8UC1 : CV_8UC3, info.ptr);

    return mat.clone();  // Clone to separate memory ownership
}