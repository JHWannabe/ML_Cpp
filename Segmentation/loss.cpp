// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "./utils/losses.hpp"

torch::Tensor CEDiceLoss::operator()(torch::Tensor& pred, torch::Tensor& target) {
    // Ensure pred is in log-softmax format
    auto log_probs = torch::log_softmax(pred, 1);
    auto ce_loss = -torch::sum(target * log_probs) / target.size(0);

    // Convert pred to probabilities
    auto probs = torch::softmax(pred, 1);

    // Dice Loss Calculation
    auto intersection = torch::sum(probs * target);
    auto dice_loss = 1 - (2.0 * intersection + 1) / (torch::sum(probs) + torch::sum(target) + 1);

    // Combined Loss
    return ce_loss + dice_loss;
}


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor &input, torch::Tensor &target){
    static auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));
    return criterion(input, target);
}
