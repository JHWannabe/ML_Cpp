#ifndef LOSSES_HPP
#define LOSSES_HPP

// For External Library
#include <torch/torch.h>

// Define Namespace
namespace nn = torch::nn;
namespace F = nn::functional;

// -------------------
// class{Loss}
// -------------------
class Loss {
public:
    Loss() {}
    torch::Tensor operator()(torch::Tensor& input, torch::Tensor& target, float dice_weight = 0.5, float focal_weight = 0.5);
    torch::Tensor l1_loss(torch::Tensor input, torch::Tensor target);
    torch::Tensor dice_loss(torch::Tensor input, torch::Tensor target, bool multiclass = false);
};

class FocalLoss : public torch::nn::Module {
public:
    float gamma = 6.0;
    torch::Tensor alpha;
    bool size_average;

    FocalLoss() : gamma(6.0), alpha(torch::Tensor()), size_average(true) {} // �⺻ ������ �߰�
    FocalLoss(float gamma, torch::Tensor alpha, bool size_average);
    torch::Tensor forward(torch::Tensor input, torch::Tensor target);
};
#endif



