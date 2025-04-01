#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <iostream>

class CosineAnnealingWarmupRestarts {
public:
    CosineAnnealingWarmupRestarts(
        torch::optim::Optimizer& optimizer,
        int first_cycle_steps,
        double cycle_mult = 1.0,
        double max_lr = 0.1,
        double min_lr = 0.001,
        int warmup_steps = 0,
        double gamma = 1.0,
        int last_epoch = -1)
        : optimizer(optimizer),
        first_cycle_steps(first_cycle_steps),
        cycle_mult(cycle_mult),
        base_max_lr(max_lr),
        max_lr(max_lr),
        min_lr(min_lr),
        warmup_steps(warmup_steps),
        gamma(gamma),
        cur_cycle_steps(first_cycle_steps),
        cycle(0),
        step_in_cycle(last_epoch)
    {
        if (warmup_steps >= first_cycle_steps) {
            throw std::invalid_argument("warmup_steps must be smaller than first_cycle_steps");
        }
        init_lr();
    }

    void step(int epoch = -1);

private:
    torch::optim::Optimizer& optimizer;
    int first_cycle_steps;
    double cycle_mult;
    double base_max_lr;
    double max_lr;
    double min_lr;
    int warmup_steps;
    double gamma;
    int cur_cycle_steps;
    int cycle;
    int step_in_cycle;
    int last_epoch = -1;
    std::vector<double> base_lrs;

    void init_lr();
    std::vector<double> get_lr();
};
