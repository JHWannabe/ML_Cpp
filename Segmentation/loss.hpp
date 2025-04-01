#ifndef LOSS_HPP
#define LOSS_HPP

// For External Library
#include <torch/torch.h>


// -----------------------------------
// class{Loss}
// -----------------------------------
class CEDiceLoss {
public:
    CEDiceLoss() {}
    torch::Tensor operator()(torch::Tensor &pred, torch::Tensor &target);
};

class Loss{
public:
    Loss(){}
    torch::Tensor operator()(torch::Tensor &input, torch::Tensor &target);
};

#endif
