"""
Training loop for the BINN.

Trainer.fit runs the four-phase schedule of training/phases.py. Each epoch:

    1. configure the phase (learning rates, frozen sub-networks, loss weights)
    2. one optimization pass over the training data
    3. one evaluation pass over the validation data
    4. phase-specific model selection (best surface in Phase 1, best model
       and early stopping in Phase 4)
    5. logging and checkpointing

One-time work at phase boundaries (restoring the best surface, measuring the
loss scales) is done in _enter_phase.
"""
import os
import time

import torch

from modules.binn_eql.equations.extract import extract_params
from modules.binn_eql.equations.format import write_equations
from modules.binn_eql.training.checkpoint import (
    load_checkpoint, load_weights, save_checkpoint, save_weights)
from modules.binn_eql.training.phases import PhaseSchedule
from modules.utils.time_remaining import time_remaining

LOSS_KEYS = ('loss', 'gls', 'pde', 'reg', 'mass')


def _empty_param_history():
    return {'raw_w_unscaled': [], 'effective_unscaled': [], 'diffusion_coeffs': [], 'epoch': []}


class BestTracker:
    """Lowest value seen so far and the epoch at which it occurred."""

    def __init__(self, value=1e12, epoch=0):
        self.value, self.epoch = value, epoch

    @classmethod
    def from_history(cls, values, offset, default_epoch):
        """Best entry of a loss history (values[k] belongs to epoch offset + k)."""
        if not values:
            return cls(epoch=default_epoch)
        best = min(values)
        return cls(best, offset + values.index(best))

    def update(self, value, epoch, rel_threshold=0.0):
        """Record value if it improves on the best by more than rel_threshold. Returns True if so."""
        if (self.value - value) / self.value > rel_threshold:
            self.value, self.epoch = value, epoch
            return True
        return False


class Trainer:
    """
    Parameters
    ----------
    model        BINN
    loss_fn      BINNLoss for the same model
    optimizer    param groups named 'surface', 'reaction' and optionally 'diffusion'
    scheduler    stepped once per epoch during Phase 1 only
    out_dir      run directory (equation log, checkpoints, best weights)
    mass_weight  weight of the mass-conservation loss
    """

    def __init__(self, model, loss_fn, optimizer, scheduler=None, out_dir='.',
                 save_prefix=None, mass_weight=1.0, log_every=1000,
                 checkpoint_every=50, floor_track_every=100):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.out_dir = out_dir
        self.save_prefix = save_prefix or os.path.join(out_dir, 'binn')
        self.mass_weight = mass_weight
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.floor_track_every = floor_track_every

        # The training script's param groups are the single source of truth
        # for LRs. Read 'max_lr' rather than 'lr': OneCycleLR overwrites
        # 'lr' with max_lr / div_factor when it is constructed.
        self.base_lrs = {pg.get('name'): pg.get('max_lr', pg['lr'])
                         for pg in optimizer.param_groups}

        self.train_loss_dict = {k: [] for k in LOSS_KEYS}
        self.val_loss_dict = {k: [] for k in LOSS_KEYS}
        self.param_history = _empty_param_history()
        self._surface_restored = False

    # ==================================================================
    # Main loop
    # ==================================================================
    def fit(self, train_data, val_data, epochs, batch_size, l0_weight=0.0,
            phase_ends=None, initial_epoch=0, early_stopping=None,
            phase1_early_stopping=None):
        """
        Train from initial_epoch up to epochs.

        phase_ends             (phase_1_end, phase_2_end, phase_3_end); default 5/20/30% of epochs
        early_stopping         Phase 4 patience, in epochs without val-loss improvement
        phase1_early_stopping  Phase 1 patience on val GLS; ends Phase 1 early
                               (later phases keep their end epochs)

        Returns (param_history, train_loss_dict, val_loss_dict).
        """
        schedule = PhaseSchedule(*phase_ends) if phase_ends else PhaseSchedule.default(epochs)
        schedule.validate(epochs)
        self._check_scheduler_length(schedule)

        if initial_epoch == 0:
            self.param_history = _empty_param_history()

        # Best surface (Phase 1) and best model (Phase 4), recovered from the
        # loss history when resuming.
        best_surface = BestTracker.from_history(
            self.val_loss_dict['gls'][:schedule.phase_1_end], 0, initial_epoch)
        best_model = BestTracker.from_history(
            self.val_loss_dict['loss'][schedule.phase_3_end:], schedule.phase_3_end,
            max(initial_epoch, schedule.phase_3_end))
        self._surface_restored = initial_epoch >= schedule.phase_1_end

        device = train_data.device
        total_iter = initial_epoch + epochs
        start_time = epoch_start = time.time()
        current_phase = None
        epoch = initial_epoch

        for epoch in range(initial_epoch, epochs):
            epoch_start = time.time()
            phase = schedule.phase_at(epoch)
            entering = phase.number != current_phase

            # --- 1. Configure the phase ---------------------------------
            if entering:
                self._enter_phase(phase.number, train_data, best_surface)
            self._configure_phase(phase)
            if entering:
                self._print_phase_banner(phase.number, epoch)
                current_phase = phase.number

            weights = torch.tensor(
                [phase.gls_weight, phase.pde_weight,
                 schedule.l0_weight(phase, epoch, l0_weight, self.loss_fn.l0_scale)],
                device=device)

            if phase.number == 2 and epoch % self.floor_track_every == 0:
                self.loss_fn.track_pde_floor(epoch)
            if epoch % self.log_every == 0:
                self._snapshot_equation(epoch)

            # --- 2-3. Train and validate --------------------------------
            self._record(self.train_loss_dict,
                         self._run_epoch(train_data, batch_size, weights, phase, train=True))
            if self.scheduler is not None and phase.steps_scheduler:
                self.scheduler.step()
            self._record(self.val_loss_dict,
                         self._run_epoch(val_data, batch_size, weights, phase, train=False))

            # --- 4. Model selection -------------------------------------
            if phase.number == 1:
                # GLS, not total loss: only the surface trains in Phase 1.
                if best_surface.update(self.val_loss_dict['gls'][-1], epoch):
                    save_weights(self.model, self._weights_path('best_phase1'))
                if (phase1_early_stopping is not None and self._scheduler_finished(epoch)
                        and epoch - best_surface.epoch >= phase1_early_stopping):
                    print(f"Phase 1 early stopping at epoch {epoch} (no val GLS "
                          f"improvement in {phase1_early_stopping} epochs)", flush=True)
                    schedule.phase_1_end = epoch + 1

            if phase.number == 4:
                if best_model.update(self.val_loss_dict['loss'][-1], epoch):
                    save_weights(self.model, self._weights_path('best_val'))
                if early_stopping is not None and epoch - best_model.epoch >= early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # --- 5. Logging and checkpointing ---------------------------
            if epoch % self.log_every == 0:
                self._print_progress(epoch, start_time, epoch_start, batch_size, total_iter)
            if epoch % self.checkpoint_every == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch)

        self._print_summary(epoch, best_model.epoch, start_time, epoch_start, batch_size, total_iter)
        return self.param_history, self.train_loss_dict, self.val_loss_dict

    # ==================================================================
    # Phases
    # ==================================================================
    def _enter_phase(self, phase_number, train_data, best_surface):
        """
        One-time work at a phase boundary. Also runs when a resumed run
        starts mid-schedule, so the loss scales are rebuilt; each call is
        a no-op if its work has already been done.
        """
        if phase_number >= 2:
            # The PDE/mass scales and the collocation cache are measured from
            # the surface and then frozen, so the best Phase-1 surface must be
            # restored before they are computed.
            if not self._surface_restored:
                path = self._weights_path('best_phase1')
                if os.path.exists(path):
                    load_weights(self.model, path, device=train_data.device)
                    print(f"Restored best Phase-1 surface from epoch {best_surface.epoch} "
                          f"(val GLS = {best_surface.value:.4e})", flush=True)
                self._surface_restored = True
            self.loss_fn.calibrate_physics(train_data)

        if phase_number >= 3:
            # Start sparsification from the best reaction Phase 2 found.
            self.loss_fn.calibrate_l0()

    def _configure_phase(self, phase):
        """
        Set every param group's LR and every sub-network's requires_grad for
        this phase.

        In a phase where the scheduler steps (Phase 1), the scheduler owns
        the LR of the groups being trained; overwriting it here every epoch
        would pin them at base LR and disable the schedule. In every other
        phase the LR is set explicitly: a group left alone would inherit
        whatever OneCycleLR annealed it to at the end of Phase 1 (~4e-9).

        Frozen groups get LR 0. The scheduler may overwrite that in Phase 1,
        but frozen parameters receive no gradients and are skipped by the
        optimizer either way.
        """
        scheduled = phase.steps_scheduler and self.scheduler is not None
        for pg in self.optimizer.param_groups:
            name = pg.get('name')
            if name not in phase.trains:
                pg['lr'] = 0.0
            elif not scheduled:
                pg['lr'] = self.base_lrs.get(name, 0.0)

        subnets = {'surface': self.model.surface_fitter,
                   'reaction': self.model.reaction,
                   'diffusion': self.model.diffusion_fitter}
        for name, net in subnets.items():
            if net is not None:
                net.requires_grad_(name in phase.trains)

    def _check_scheduler_length(self, schedule):
        steps = getattr(self.scheduler, 'total_steps', None)
        if steps is not None and steps != schedule.phase_1_end:
            print(f"WARNING: scheduler total_steps={steps} != phase_1_end="
                  f"{schedule.phase_1_end}. The scheduler only steps during Phase 1, "
                  f"so Phase 1 will end mid-ramp (or the ramp will finish early).", flush=True)

    def _scheduler_finished(self, epoch):
        """
        True once OneCycleLR is 90% through. Validation loss rises at peak LR
        and falls during the anneal, so patience-based stopping before then
        tends to fire just before the improvement it is waiting for.
        """
        if self.scheduler is None:
            return True
        return epoch >= int(0.9 * getattr(self.scheduler, 'total_steps', 0))

    # ==================================================================
    # One pass over the data
    # ==================================================================
    def _run_epoch(self, data, batch_size, weights, phase, train):
        """One pass over data (with optimizer steps if train). Returns per-sample mean losses."""
        self.model.train(train)
        species = self.model.species
        order = torch.randperm(len(data)) if train else torch.arange(len(data))
        totals = dict.fromkeys(LOSS_KEYS, 0.0)

        # Gradients stay enabled during validation: the uncached collocation
        # path differentiates the surface with respect to its inputs.
        for start in range(0, len(data), batch_size):
            batch = data[order[start:start + batch_size]]
            inputs, targets = batch[:, :-species], batch[:, -species:]

            if train:
                self.optimizer.zero_grad()

            terms = self.loss_fn(inputs, self.model(inputs), targets, phase.number)
            weighted = torch.stack([terms.gls, terms.pde, terms.l0]) * weights
            mass = (self.mass_weight * terms.mass if phase.mass_active
                    else torch.tensor(0.0, device=inputs.device))
            total = weighted.sum() + terms.soft_wall + mass

            if train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            n = len(inputs)
            totals['loss'] += total.item() * n
            totals['gls'] += weighted[0].item() * n
            totals['pde'] += weighted[1].item() * n
            totals['reg'] += (weighted[2] + terms.soft_wall).item() * n
            totals['mass'] += mass.item() * n

        return {k: v / len(data) for k, v in totals.items()}

    @staticmethod
    def _record(history, losses):
        for k, v in losses.items():
            history[k].append(v)

    # ==================================================================
    # Logging
    # ==================================================================
    def _diffusion_values(self):
        return [D.item() for D in self.model.diffusion_fitter()]

    def _snapshot_equation(self, epoch):
        """Append the current equation to equation.txt and record coefficients in param_history."""
        path = os.path.join(self.out_dir, 'equation.txt')
        write_equations(path, self.model, header=f'Equation Epoch {epoch}')
        with open(path, 'a') as f:
            f.write('\n')
            if self.model.learns_diffusion:
                f.write(f'{self._diffusion_values()}\n\n')

        snapshot = extract_params(self.model, full=False)
        self.param_history['epoch'].append(epoch)
        self.param_history['raw_w_unscaled'].append([p['raw_w_unscaled'] for p in snapshot])
        self.param_history['effective_unscaled'].append([p['effective_unscaled'] for p in snapshot])
        if self.model.learns_diffusion:
            self.param_history['diffusion_coeffs'].append(self._diffusion_values())

    def _print_phase_banner(self, phase_number, epoch):
        lrs = ", ".join(f"{pg.get('name')}={pg['lr']:.2e}" for pg in self.optimizer.param_groups)
        print(f"=== Phase {phase_number} @ epoch {epoch} | lrs: {lrs} ===\n", flush=True)

    def _print_progress(self, epoch, start_time, epoch_start, batch_size, total_iter):
        _, remaining, _ = time_remaining(
            current_iter=epoch + 1, total_iter=total_iter, start_time=start_time,
            previous_time=epoch_start, ops_per_iter=batch_size)
        train, val = self.train_loss_dict, self.val_loss_dict

        line = (f"Epoch {epoch} | Train loss = {train['loss'][-1]:.4e}"
                f" | Val loss = {val['loss'][-1]:.4e}"
                f" | Val GLS = {val['gls'][-1]:.4e}, Val PDE = {val['pde'][-1]:.4e},"
                f" Val Reg = {val['reg'][-1]:.4e} Val Mass = {val['mass'][-1]:.4e}")
        if self.model.learns_diffusion:
            line += ' | ' + ', '.join(f'D{name}={D:.4e}' for name, D in
                                      zip(self.model.species_names, self._diffusion_values()))
        print(f"{line} | Remaining = {remaining}           ", flush=True)

    def _print_summary(self, epoch, best_epoch, start_time, epoch_start, batch_size, total_iter):
        """Final line: the losses of the selected (best Phase-4) model."""
        elapsed, _, _ = time_remaining(
            current_iter=epoch + 1, total_iter=total_iter, start_time=start_time,
            previous_time=epoch_start, ops_per_iter=batch_size)
        i = best_epoch if best_epoch < len(self.val_loss_dict['loss']) else -1
        train, val = self.train_loss_dict, self.val_loss_dict
        print(f"Epoch {epoch} | Train loss = {train['loss'][i]:.4e}"
              f" | Val loss = {val['loss'][i]:.4e}"
              f" | Val GLS = {val['gls'][i]:.4e}, Val PDE = {val['pde'][i]:.4e},"
              f" Val Reg = {val['reg'][i]:.4e} | Elapsed = {elapsed}           ", flush=True)

    # ==================================================================
    # Persistence
    # ==================================================================
    def _weights_path(self, tag):
        return f"{self.save_prefix}_{tag}_model"

    def _save_checkpoint(self, epoch):
        save_checkpoint(
            os.path.join(self.out_dir, 'latest_checkpoint.pt'),
            epoch=epoch, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            train_losses=self.train_loss_dict, val_losses=self.val_loss_dict,
            param_history=self.param_history, loss_state=self.loss_fn.state_dict())

    def resume(self, path, device='cuda'):
        """Restore a checkpoint written by this Trainer. Returns the epoch to resume from."""
        epoch, checkpoint = load_checkpoint(path, self.model, self.optimizer, self.scheduler, device)
        self.train_loss_dict = checkpoint['train_loss_dict']
        self.val_loss_dict = checkpoint['val_loss_dict']
        self.param_history = checkpoint['param_history']
        if 'loss_state' in checkpoint:
            self.loss_fn.load_state_dict(checkpoint['loss_state'])
        else:
            print("Checkpoint predates saved loss state: if resuming past Phase 2, "
                  "the L0 scale will be re-measured from the current weights.", flush=True)
        return epoch

    def load(self, path, device=None):
        """Load model weights (e.g. binn_best_val_model) and switch to eval mode."""
        load_weights(self.model, path, device=device)
        self.model.eval()

    def predict(self, inputs):
        self.model.eval()
        return self.model(inputs)