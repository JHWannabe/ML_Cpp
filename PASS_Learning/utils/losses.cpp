#include <cmath>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "losses.hpp"

// Define Namespace
namespace F = nn::functional;

// -----------------------------------
// class{Loss} -> operator
// -----------------------------------
torch::Tensor Loss::operator()(torch::Tensor& output, torch::Tensor& target, float weight1, float weight2) {
    torch::Tensor scores = torch::cat({ 1 - output, output }, 1);
    FocalLoss focal;
    torch::Tensor l1_value = l1_loss(output.squeeze(1), target);
    torch::Tensor focal_value = focal.forward(scores, target);
    //std::cout << weight1 << " * " << l1_value << " + " << weight2 << " * " << focal_value << std::endl;
    return weight1 * l1_value + weight2 * focal_value; 
}

// -----------------------------------
// L1 Loss Function
// -----------------------------------
torch::Tensor Loss::l1_loss(torch::Tensor input, torch::Tensor target)
{
    // L1 Loss
    static auto l1_loss = nn::L1Loss(nn::L1LossOptions().reduction(torch::kMean));
    return l1_loss(input, target);
}


// -----------------------------------
// Focal Loss Function
// -----------------------------------
FocalLoss::FocalLoss(float gamma, torch::Tensor alpha, bool size_average)
    : gamma(gamma), alpha(alpha), size_average(size_average) {
    if (alpha.dim() == 0) {
        alpha = torch::cat({ alpha.unsqueeze(0), (1 - alpha).unsqueeze(0) });
    }
}

torch::Tensor FocalLoss::forward(torch::Tensor input, torch::Tensor target) {
    try {
        if (input.dim() > 2) {
            input = input.view({ input.size(0), input.size(1), -1 });
            input = input.transpose(1, 2);
            input = input.contiguous().view({ -1, input.size(2) });
        }

        // Ensure target is int64 and correctly shaped
        target = target.to(torch::kInt64);
        target = target.view({ -1, 1 });

        // Ensure target values are within range (debugging purpose)
        target = target.clamp(0, input.size(1) - 1);

        // Compute log softmax
        torch::Tensor logpt = torch::log_softmax(input, 1);
        torch::Tensor logpt_ = logpt.gather(1, target);
        logpt_ = logpt_.view({ -1 });
        torch::Tensor pt = logpt_.exp();

        // Apply alpha weighting if defined
        if (alpha.defined()) {
            if (alpha.scalar_type() != input.scalar_type()) {
                alpha = alpha.to(input.device(), input.scalar_type());
            }
            torch::Tensor at = alpha.gather(0, target);
            logpt_ = logpt_ * at;
        }

        // Compute focal loss
        torch::Tensor loss = -1 * torch::pow(1 - pt, gamma) * logpt_;
        return size_average ? loss.mean() : loss.sum();
    }
    catch (const std::exception& e) {
        std::cerr << "Error in FocalLoss::forward: " << e.what() << std::endl;
        return torch::Tensor(); // Return empty tensor on error
    }
}



// -----------------------------------
// Dice Loss Function
// -----------------------------------
float dice_coeff(torch::Tensor input, torch::Tensor target, bool reduce_batch_first = false, float epsilon = 1e-6) {
    // Check the input dimensions
    assert(input.sizes() == target.sizes());
    assert(input.dim() == 3 || !reduce_batch_first);

    // Sum dimension based on input dimensions
    std::vector<int64_t> sum_dim = (input.dim() == 2 || !reduce_batch_first) ? std::vector<int64_t>{-1, -2} : std::vector<int64_t>{ -1, -2, -3 };

    // Compute intersection (2 * sum of element-wise multiplication)
    torch::Tensor inter = 2 * (input * target).sum(sum_dim);

    // Compute the sum of sets (input + target)
    torch::Tensor sets_sum = input.sum(sum_dim) + target.sum(sum_dim);

    // Avoid division by zero
    sets_sum = torch::where(sets_sum == 0, inter, sets_sum);

    // Calculate Dice coefficient
    torch::Tensor dice = (inter + epsilon) / (sets_sum + epsilon);

    // Return mean Dice coefficient
    return dice.mean().item<float>();
}

float multiclass_dice_coeff(torch::Tensor input, torch::Tensor target, bool reduce_batch_first = false, float epsilon = 1e-6) {
    // Flatten the input and target tensors and compute the Dice coefficient
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon);
}

torch::Tensor Loss::dice_loss(torch::Tensor input, torch::Tensor target, bool multiclass) {
    // Dice loss (objective to minimize) between 0 and 1
    float (*fn)(torch::Tensor, torch::Tensor, bool, float) = multiclass ? multiclass_dice_coeff : dice_coeff;
    return 1 - torch::tensor(fn(input, target, true, 1e-6));
}