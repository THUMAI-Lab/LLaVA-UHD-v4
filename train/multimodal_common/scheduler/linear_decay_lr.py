""" Linear Decay Scheduler

Linear LR schedule with warmup.

Copyright @2026 modelbest
"""
import logging
import math
import torch
from timm.scheduler.scheduler import Scheduler

_logger = logging.getLogger(__name__)


class LinearDecayLRScheduler(Scheduler):
    """
    Linear decay learning rate scheduler with warmup.
    The learning rate decays linearly from the initial learning rate to a minimum learning rate
    over a specified number of steps.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            warmup_t: int = 0,
            warmup_lr_init: float = 0.,
            warmup_prefix: bool = False,
            t_in_epochs: bool = True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            initialize: bool = True,
    ) -> None:
        """
        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            t_initial (int): Number of steps for the scheduler to complete one cycle.
            lr_min (float): Minimum learning rate.
            warmup_t (int): Number of warmup steps.
            warmup_lr_init (float): The initial learning rate for warmup.
            warmup_prefix (bool): If True, the warmup period is considered part of the total schedule time `t_initial`.
            t_in_epochs (bool): If True, `t_initial` and `warmup_t` are in epochs, otherwise in steps.
            noise_range_t (tuple): Range of steps to apply noise.
            noise_pct (float): Percentage of noise to apply.
            noise_std (float): Standard deviation of noise.
            noise_seed (int): Seed for noise generation.
            initialize (bool): If True, initialize the scheduler.
        """
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0, "t_initial must be positive"
        assert lr_min >= 0, "lr_min must be non-negative"
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix

        if self.warmup_t > 0:
            self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = []

    def _get_lr(self, t):
        if t < self.warmup_t:
            # Linear warmup from warmup_lr_init to base_values
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # Linear decay from base_values to lr_min
            t_decay = t
            if self.warmup_prefix:
                t_decay = t - self.warmup_t

            total_decay_steps = self.t_initial - self.warmup_t
            if total_decay_steps <= 0:
                 # If no decay steps are specified after warmup, hold the base LR
                return self.base_values
            
            progress = min(1.0, t_decay / total_decay_steps)
            
            lrs = [v - (v - self.lr_min) * progress for v in self.base_values]
            
        return lrs

    def get_cycle_length(self, cycles=0):
        """
        This scheduler does not support cycles, but this method is implemented for API compatibility.
        It returns the total length of the schedule.
        """
        return self.t_initial 