#include "scheduler.hpp"

void CosineAnnealingWarmupRestarts::step(int epoch) {
    if (epoch == -1) {
        epoch = last_epoch + 1;
        step_in_cycle++;
        if (step_in_cycle >= cur_cycle_steps) {
            cycle++;
            step_in_cycle -= cur_cycle_steps;
            cur_cycle_steps = static_cast<int>((cur_cycle_steps - warmup_steps) * cycle_mult) + warmup_steps;
        }
    }
    else {
        if (epoch >= first_cycle_steps) {
            if (cycle_mult == 1.0) {
                step_in_cycle = epoch % first_cycle_steps;
                cycle = epoch / first_cycle_steps;
            }
            else {
                int n = static_cast<int>(std::log(epoch / first_cycle_steps * (cycle_mult - 1) + 1) / std::log(cycle_mult));
                cycle = n;
                step_in_cycle = epoch - static_cast<int>(first_cycle_steps * (std::pow(cycle_mult, n) - 1) / (cycle_mult - 1));
                cur_cycle_steps = static_cast<int>(first_cycle_steps * std::pow(cycle_mult, n));
            }
        }
        else {
            cur_cycle_steps = first_cycle_steps;
            step_in_cycle = epoch;
        }
    }

    max_lr = base_max_lr * std::pow(gamma, cycle);
    last_epoch = epoch;

    auto new_lrs = get_lr();
    auto& param_groups = optimizer.param_groups();
    for (size_t i = 0; i < param_groups.size(); ++i) {
        param_groups[i].options().set_lr(new_lrs[i]);
    }
}


void CosineAnnealingWarmupRestarts::init_lr() {
    base_lrs.clear();
    for (auto& param_group : optimizer.param_groups()) {
        double lr = min_lr;
        param_group.options().set_lr(lr);
        base_lrs.push_back(lr);
    }
}

std::vector<double> CosineAnnealingWarmupRestarts::get_lr() {
    std::vector<double> lrs;
    if (step_in_cycle == -1) {
        return base_lrs;
    }
    else if (step_in_cycle < warmup_steps) {
        for (double base_lr : base_lrs) {
            lrs.push_back((max_lr - base_lr) * step_in_cycle / warmup_steps + base_lr);
        }
    }
    else {
        for (double base_lr : base_lrs) {
            lrs.push_back(base_lr + (max_lr - base_lr) *
                (1 + std::cos(M_PI * (step_in_cycle - warmup_steps) / (cur_cycle_steps - warmup_steps))) / 2);
        }
    }
    return lrs;
}