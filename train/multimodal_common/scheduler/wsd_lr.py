""" WSD (Warmup-Stable-Decay) Scheduler

WSD LR schedule with warmup, stable, and decay phases.

Copyright @2026 modelbest
"""
import logging
import torch
from timm.scheduler.scheduler import Scheduler

_logger = logging.getLogger(__name__)


class WSDLRScheduler(Scheduler):
    """
    WSD (Warmup-Stable-Decay) learning rate scheduler.
    
    The learning rate follows three phases:
    1. Warmup phase (0 to warmup_t): linearly increases from warmup_lr_init to lr_max
    2. Stable phase (warmup_t to stable_t): remains constant at lr_max
    3. Decay phase (stable_t to t_initial): linearly decays from lr_max to lr_min
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            stable_t: int,
            lr_min: float = 0.,
            warmup_t: int = 0,
            warmup_lr_init: float = 0.,
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
            t_initial (int): Total number of steps for the scheduler to complete.
            stable_t (int): Step at which the stable phase ends and decay begins.
            lr_min (float): Minimum learning rate at the end of decay phase.
            warmup_t (int): Number of warmup steps.
            warmup_lr_init (float): Initial learning rate for warmup phase.
            t_in_epochs (bool): If True, `t_initial`, `stable_t` and `warmup_t` are in epochs, otherwise in steps.
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
        assert stable_t >= warmup_t, "stable_t must be greater than or equal to warmup_t"
        assert stable_t <= t_initial, "stable_t must be less than or equal to t_initial"
        assert lr_min >= 0, "lr_min must be non-negative"
        
        self.t_initial = t_initial
        self.stable_t = stable_t
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init

        # Calculate warmup step size for each param group
        if self.warmup_t > 0:
            self.warmup_steps = [(v - self.warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = []

    def _get_lr(self, t):
        if t < self.warmup_t:
            # Phase 1: Warmup - linear increase from warmup_lr_init to lr_max (base_values)
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        elif t < self.stable_t:
            # Phase 2: Stable - keep at lr_max (base_values)
            lrs = self.base_values
        elif t < self.t_initial:
            # Phase 3: Decay - linear decay from lr_max to lr_min
            total_decay_steps = self.t_initial - self.stable_t
            if total_decay_steps <= 0:
                # If no decay steps, hold the base LR
                return self.base_values
            
            t_decay = t - self.stable_t
            progress = min(1.0, t_decay / total_decay_steps)
            
            lrs = [v - (v - self.lr_min) * progress for v in self.base_values]
        else:
            # Phase 4: After t_initial - keep at lr_min
            lrs = [self.lr_min for _ in self.base_values]
            
        return lrs

    def get_cycle_length(self, cycles=0):
        """
        This scheduler does not support cycles, but this method is implemented for API compatibility.
        It returns the total length of the schedule.
        """
        return self.t_initial