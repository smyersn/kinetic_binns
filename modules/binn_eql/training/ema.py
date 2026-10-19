"""
Exponential moving average (EMA) of a sub-network's weights.

Used for the surface fitter in Phase 1. At a constant learning rate the
weights keep jittering around a good solution, partly following the noise in
each minibatch; their running average cancels much of that jitter, so the
averaged surface usually generalizes better than any single snapshot.
Training itself is unchanged: the optimizer keeps updating the ordinary
weights, and the average only follows them.

    ema = WeightEMA(model.surface_fitter, decay=0.999)
    ...                  # after every optimizer step:
    ema.update()
    with ema.applied():  # evaluate / save with the averaged weights
        ...
"""
from contextlib import contextmanager

import torch


class WeightEMA:
    """
    Running average of every floating-point parameter of `module`.

    decay   weight kept by the average at each update; the average spans
            roughly 1 / (1 - decay) updates.
    warmup  for the first updates the effective decay is
            min(decay, (1 + n) / (10 + n)), so the average is not dominated
            by the random initial weights while it fills.
    """

    def __init__(self, module, decay, warmup=True):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.module = module
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self.shadow = {name: p.detach().clone() for name, p in self._params()}
        self._backup = None

    def _params(self):
        return [(n, p) for n, p in self.module.named_parameters() if p.dtype.is_floating_point]

    def _current_decay(self):
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self):
        """Fold the module's current weights into the average."""
        weight = 1.0 - self._current_decay()
        for name, p in self._params():
            self.shadow[name].lerp_(p.detach(), weight)
        self.num_updates += 1

    @contextmanager
    def applied(self):
        """Temporarily load the averaged weights into the module."""
        if self._backup is not None:
            raise RuntimeError("EMA weights are already applied")
        params = self._params()
        with torch.no_grad():
            self._backup = {name: p.detach().clone() for name, p in params}
            for name, p in params:
                p.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in params:
                    p.copy_(self._backup[name])
            self._backup = None

    def state_dict(self):
        return {'decay': self.decay, 'num_updates': self.num_updates,
                'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state):
        self.num_updates = state['num_updates']
        for name, p in self._params():
            if name in state['shadow']:
                self.shadow[name] = state['shadow'][name].to(p.device, p.dtype).clone()