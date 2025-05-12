// For External Library
#include <torch/torch.h>
// For Original Header
#include "loss.hpp"
#include "./utils/losses.hpp"


// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor& input, torch::Tensor& target) {
    // 1) Ÿ�� 2D�� ��� �ε����� ��ȯ
    torch::Tensor tgt_idx;
    if (target.dim() > 1) {
        // one-hot �� [batch] long
        tgt_idx = std::get<0>(torch::max(target, /*dim=*/1)).to(torch::kLong);
    }
    else {
        // �̹� [batch] ���̸� �״�� long���� ĳ����
        tgt_idx = target.to(torch::kLong);
    }

    // 2) �Է��� raw logits�̶�� �α� Ȯ���� ��ȯ
    auto log_probs = torch::log_softmax(input, /*dim=*/1);

    // 3) NLLLoss ���
    static auto criterion = torch::nn::NLLLoss(
        torch::nn::NLLLossOptions()
        .ignore_index(-100)
        .reduction(torch::kMean)
    );

    return criterion(log_probs, tgt_idx);
}