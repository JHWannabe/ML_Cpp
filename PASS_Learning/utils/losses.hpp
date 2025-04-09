#ifndef LOSSES_HPP
#define LOSSES_HPP

// For External Library
#include <torch/torch.h>

// Define Namespace
namespace nn = torch::nn;
namespace F = nn::functional;

 //-------------------
 //class{Loss}
 //-------------------
class Loss {
public:
    Loss() {}
    torch::Tensor operator()(torch::Tensor& output, torch::Tensor& target, float l1_weight = 0.6, float focal_weight = 0.4);
    torch::Tensor focalLoss(torch::Tensor output, torch::Tensor target);
};

class FocalLoss : public torch::nn::Module {
public:
    float gamma = 6.0;
    torch::Tensor alpha;
    bool size_average;

    FocalLoss() : gamma(6.0), alpha(torch::Tensor()), size_average(true) {} // 기본 생성자 추가
    FocalLoss(float gamma, torch::Tensor alpha, bool size_average);
    torch::Tensor forward(torch::Tensor input, torch::Tensor target);
};

#endif