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
torch::Tensor Loss::operator()(torch::Tensor& output, torch::Tensor& target, float l1_weight, float focal_weight) {
    torch::Tensor s_output = output.squeeze(1);
    torch::Tensor l1_value = l1_loss(s_output, target);
    torch::Tensor scores = torch::cat({ 1 - output, output }, 1);
    torch::Tensor focal_value = this->focalLoss(scores, target);
    return l1_weight * l1_value + focal_weight * focal_value;
}

// -----------------------------------
// Focal Loss Function
// -----------------------------------
torch::Tensor Loss::focalLoss(torch::Tensor output, torch::Tensor target)
{
	// Focal Loss
	static auto focal_loss = FocalLoss(6.0, torch::Tensor(), true);
	return focal_loss.forward(output, target);
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

torch::Tensor FocalLoss::forward(torch::Tensor output, torch::Tensor target) {
    try {
        if (output.dim() > 2) {
            output = output.view({ output.size(0), output.size(1), -1 });
            output = output.transpose(1, 2);
            output = output.contiguous().view({ -1, output.size(2) });
        }

        // Ensure target is int64 and correctly shaped
        target = target.to(torch::kInt64);
        target = target.view({ -1, 1 });

        // Ensure target values are within range (debugging purpose)
        target = target.clamp(0, output.size(1) - 1);

        // Compute log softmax
        torch::Tensor logpt = torch::log_softmax(output, 1);
        torch::Tensor logpt_ = logpt.gather(1, target);
        logpt_ = logpt_.view({ -1 });
        torch::Tensor pt = logpt_.exp();

        // Apply alpha weighting if defined
        if (alpha.defined()) {
            if (alpha.scalar_type() != output.scalar_type()) {
                alpha = alpha.to(output.device(), output.scalar_type());
            }
            torch::Tensor at = alpha.gather(0, target.data().view(-1));
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
float dice_coeff(torch::Tensor output, torch::Tensor target, bool reduce_batch_first = false, float epsilon = 1e-6) {
    // Check the input dimensions
    assert(output.sizes() == target.sizes());
    assert(output.dim() == 3 || !reduce_batch_first);

    // Sum dimension based on input dimensions
    std::vector<int64_t> sum_dim = (output.dim() == 2 || !reduce_batch_first) ? std::vector<int64_t>{-1, -2} : std::vector<int64_t>{ -1, -2, -3 };

    // Compute intersection (2 * sum of element-wise multiplication)
    torch::Tensor inter = 2 * (output * target).sum(sum_dim);

    // Compute the sum of sets (input + target)
    torch::Tensor sets_sum = output.sum(sum_dim) + target.sum(sum_dim);

    // Avoid division by zero
    sets_sum = torch::where(sets_sum == 0, inter, sets_sum);

    // Calculate Dice coefficient
    torch::Tensor dice = (inter + epsilon) / (sets_sum + epsilon);

    // Return mean Dice coefficient
    return dice.mean().item<float>();
}

float multiclass_dice_coeff(torch::Tensor output, torch::Tensor target, bool reduce_batch_first = false, float epsilon = 1e-6) {
    // Flatten the input and target tensors and compute the Dice coefficient
    return dice_coeff(output.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon);
}
