#include "diffusion.h"

// 전역으로 한 번만 생성
static std::unique_ptr<py::scoped_interpreter> g_interpreter;

void initializePython() {
    if (!g_interpreter) {
        g_interpreter = std::make_unique<py::scoped_interpreter>();
        // 필요하다면 여기서 GIL 해제
        // py::gil_scoped_release release;
    }
}

cv::Mat stableDiffusion(std::string& file_path) {
    // 인터프리터가 안 띄워져 있으면 바로 초기화
    if (!g_interpreter)
        initializePython();

    // GIL 확보
    py::gil_scoped_acquire gil;

    // 스크립트 경로 우선 추가
    auto sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "../PASS_Learning/");

    // 모듈 로드 & 함수 호출
    auto sd_module = py::module_::import("stableDiffusion");
    auto array = sd_module
        .attr("anomaly_source")(file_path)
        .cast<py::array_t<uint8_t>>();

    // numpy → cv::Mat 변환
    auto info = array.request();
    int h = info.shape[0], w = info.shape[1];
    int c = (info.ndim == 3 ? info.shape[2] : 1);
    cv::Mat mat(h, w, c == 1 ? CV_8UC1 : CV_8UC3, info.ptr);

    return mat.clone();  // 메모리 소유권 분리
}