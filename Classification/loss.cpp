// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "./utils/losses.hpp"


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor& input, torch::Tensor& target) {
    // 1) 타깃 2D인 경우 인덱스로 변환
    torch::Tensor tgt_idx;
    if (target.dim() > 1) {
        // one-hot → [batch] long
        tgt_idx = std::get<0>(torch::max(target, /*dim=*/1)).to(torch::kLong);
    }
    else {
        // 이미 [batch] 라벨이면 그대로 long으로 캐스팅
        tgt_idx = target.to(torch::kLong);
    }

    // 2) 입력이 raw logits이라면 로그 확률로 변환
    auto log_probs = torch::log_softmax(input, /*dim=*/1);

    // 3) NLLLoss 계산
    static auto criterion = torch::nn::NLLLoss(
        torch::nn::NLLLossOptions()
        .ignore_index(-100)
        .reduction(torch::kMean)
    );

    return criterion(log_probs, tgt_idx);
}