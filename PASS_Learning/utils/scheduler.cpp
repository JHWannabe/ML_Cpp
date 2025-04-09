#include "scheduler.hpp"

void CosineAnnealingWarmupRestarts::step(int epoch) {
   if (epoch == -1) {
       epoch = this->last_epoch + 1;
       this->step_in_cycle++;
       if (this->step_in_cycle >= this->cur_cycle_steps) {
           this->cycle++;
           this->step_in_cycle -= this->cur_cycle_steps;
           this->cur_cycle_steps = static_cast<int>((this->cur_cycle_steps - this->warmup_steps) * this->cycle_mult) + this->warmup_steps;
       }
   }
   else {
       if (epoch >= this->first_cycle_steps) {
           if (this->cycle_mult == 1.0) {
               this->step_in_cycle = epoch % this->first_cycle_steps;
               this->cycle = epoch / this->first_cycle_steps;
           }
           else {
               int n = static_cast<int>(std::log(epoch / this->first_cycle_steps * (this->cycle_mult - 1) + 1) / std::log(this->cycle_mult));
               this->cycle = n;
               this->step_in_cycle = epoch - static_cast<int>(this->first_cycle_steps * (std::pow(this->cycle_mult, n) - 1) / (this->cycle_mult - 1));
               this->cur_cycle_steps = static_cast<int>(this->first_cycle_steps * std::pow(this->cycle_mult, n));
           }
       }
       else {
           this->cur_cycle_steps = this->first_cycle_steps;
           this->step_in_cycle = epoch;
       }
   }

   this->max_lr = this->base_max_lr * std::pow(this->gamma, this->cycle);
   this->last_epoch = epoch;

   auto new_lrs = get_lr();
   auto& param_groups = optimizer.param_groups();
   for (size_t i = 0; i < param_groups.size(); ++i) {
       param_groups[i].options().set_lr(new_lrs[i]);
   }
}


void CosineAnnealingWarmupRestarts::init_lr() {
   this->base_lrs.clear();
   for (auto& param_group : optimizer.param_groups()) {
       double lr = this->min_lr;
       param_group.options().set_lr(lr);
       this->base_lrs.push_back(lr);
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