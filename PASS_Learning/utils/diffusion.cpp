#include "diffusion.h"

// �������� �� ���� ����
static std::unique_ptr<py::scoped_interpreter> g_interpreter;

void initializePython() {
    if (!g_interpreter) {
        g_interpreter = std::make_unique<py::scoped_interpreter>();
        // �ʿ��ϴٸ� ���⼭ GIL ����
        // py::gil_scoped_release release;
    }
}

cv::Mat stableDiffusion(std::string& file_path) {
    // ���������Ͱ� �� ����� ������ �ٷ� �ʱ�ȭ
    if (!g_interpreter)
        initializePython();

    // GIL Ȯ��
    py::gil_scoped_acquire gil;

    // ��ũ��Ʈ ��� �켱 �߰�
    auto sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "../PASS_Learning/");

    // ��� �ε� & �Լ� ȣ��
    auto sd_module = py::module_::import("stableDiffusion");
    auto array = sd_module
        .attr("anomaly_source")(file_path)
        .cast<py::array_t<uint8_t>>();

    // numpy �� cv::Mat ��ȯ
    auto info = array.request();
    int h = info.shape[0], w = info.shape[1];
    int c = (info.ndim == 3 ? info.shape[2] : 1);
    cv::Mat mat(h, w, c == 1 ? CV_8UC1 : CV_8UC3, info.ptr);

    return mat.clone();  // �޸� ������ �и�
}