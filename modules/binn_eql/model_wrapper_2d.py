import os
import torch, time
import numpy as np

from modules.utils.time_remaining import *
from modules.utils.gradient import gradient

# --- ENFORCE TRUE FP32 PRECISION ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class model_wrapper():
    # ==================================================================
    # PHASE CONFIGURATION
    # ==================================================================
    # Each phase differs ONLY in which sub-networks train, which loss
    # terms are active, and whether the scheduler steps. Declared once
    # here instead of four near-identical if/elif blocks.
    #
    # Learning rates are NOT specified here -- they come from the
    # param_groups built in the training script, captured once as
    # self.base_lrs in __init__. A group that trains in a given phase
    # gets its base LR; a frozen group gets 0.0. The training script
    # stays the single source of truth for LR values.
    #
    # WHY LR MUST BE SET EVERY PHASE: Phase 2 previously set only
    # requires_grad and never touched lr, so it silently inherited
    # whatever OneCycleLR left behind when it finished annealing at the
    # end of Phase 1 -- max_lr/(div_factor * final_div_factor)
    # = 1e-3/2.5e5 = 4e-9 for the reaction group. Phase 2 therefore
    # trained the reaction network at ~4e-9 for its entire duration
    # (flat PDE loss for thousands of epochs), and Phase 3 -- which DID
    # set lr -- reset it to 1e-3, at which point the loss immediately
    # collapsed. Setting lr explicitly for every group every phase
    # prevents that whole class of bug.
    PHASE_CONFIG = {
        1: {  # Data only: fit the surface
            'train': ('surface',),
            'weights': (1.0, 0.0, 0.0),          # (gls, pde, l0)
            'scheduler': True,
            'mass_active': False,
        },
        2: {  # Physics on, no regularization: fit the reaction
            'train': ('reaction', 'diffusion'),
            'weights': (0.0, 1.0, 0.0),
            'scheduler': False,
            'mass_active': True,
        },
        3: {  # Ramp L0 in linearly across the phase
            'train': ('reaction', 'diffusion'),
            'weights': (0.0, 1.0, 'l0_ramp'),
            'scheduler': False,
            'mass_active': True,
        },
        4: {  # Full L0
            'train': ('reaction', 'diffusion'),
            'weights': (0.0, 1.0, 'l0_full'),
            'scheduler': False,
            'mass_active': True,
        },
    }

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 dir_name,
                 regularizer=None,
                 scheduler=None,
                 save_name=None,
                 save_best_train=False,
                 save_best_val=True,
                 save_opt=False,
                 save_reg=False,
                 mass_weight=1.0):

        self.model = model
        self.species = self.model.species
        self.optimizer = optimizer
        self.loss = loss
        self.dir_name = dir_name
        self.regularizer = regularizer
        self.scheduler = scheduler
        self.save_name = save_name
        self.save_best_train = save_best_train
        self.save_best_val = save_best_val
        self.save_opt = save_opt
        self.save_reg = save_reg
        self.mass_weight = mass_weight
        self.train_loss_dict = {'loss': [], 'gls': [], 'pde': [], 'reg': [], 'mass': []}
        self.val_loss_dict = {'loss': [], 'gls': [], 'pde': [], 'reg': [], 'mass': []}
        self.train = False
        self.val = False

        # Single source of truth for LR: whatever the training script put
        # in param_groups. _apply_phase restores these for active groups
        # so no phase can silently inherit an annealed or stale LR.
        #
        # Read max_lr, not lr: OneCycleLR overwrites pg['lr'] with
        # max_lr/div_factor at ITS construction time, and the training
        # script builds the scheduler BEFORE this wrapper -- so pg['lr']
        # here is already the scaled-down warmup value (1e-3/25 = 4e-5),
        # not the LR you asked for. PyTorch stashes the original in
        # pg['max_lr'].
        self.base_lrs = {pg.get('name'): pg.get('max_lr', pg['lr'])
                         for pg in self.optimizer.param_groups}
        print(f"Base LRs captured: {self.base_lrs}", flush=True)
        self._last_phase = None
        self._phase1_restored = False

        if self.save_name is None:
            self.save_best_train = False
            self.save_best_val = False
            self.save_opt = False
            self.save_reg = False

    # ==================================================================
    # PHASE HELPERS
    # ==================================================================
    def _subnets(self):
        """Name -> module map. diffusion_fitter is None when diff_coeffs
        are fixed, so callers must skip None."""
        return {
            'surface': self.model.surface_fitter,
            'reaction': self.model.reaction,
            'diffusion': self.model.diffusion_fitter,
        }

    def _apply_phase(self, phase, epoch, l0_weight, phase_bounds, device):
        """
        Applies a phase's optimizer settings every epoch and returns its
        (gls, pde, l0) loss weights.
        """
        cfg = self.PHASE_CONFIG[phase]

        # --- learning rates: base LR if this group trains, else 0.0 ---
        for pg in self.optimizer.param_groups:
            name = pg.get('name')
            pg['lr'] = self.base_lrs.get(name, 0.0) if name in cfg['train'] else 0.0

        # --- requires_grad ---
        for name, net in self._subnets().items():
            if net is None:
                continue
            flag = name in cfg['train']
            for p in net.parameters():
                p.requires_grad = flag

        # --- loss weights (resolve the symbolic l0 entries) ---
        gls_w, pde_w, l0_spec = cfg['weights']
        if l0_spec == 'l0_ramp':
            l0_eff = l0_weight * getattr(self.model, 'l0_scale', 1.0)
            _, phase_2_end, phase_3_end = phase_bounds
            progress = (epoch - phase_2_end) / max(phase_3_end - phase_2_end, 1)
            l0_w = l0_eff * progress
        elif l0_spec == 'l0_full':
            l0_w = l0_weight * getattr(self.model, 'l0_scale', 1.0)
        else:
            l0_w = float(l0_spec)

        # if l0_spec in ('l0_ramp', 'l0_full') and epoch % 1000 == 0:
        #     print(f"[ph{phase} ep{epoch}] l0_weight={l0_weight!r} "
        #           f"l0_scale={getattr(self.model, 'l0_scale', 'MISSING')!r} "
        #           f"-> base_weights[2]={l0_w:.4e}", flush=True)

        return torch.tensor([gls_w, pde_w, l0_w], device=device)

    def _on_phase_entry(self, phase, epoch, train_data,
                        best_phase1_idx, best_phase1_val):
        """
        One-time work at a phase boundary. Kept separate from
        _apply_phase because these must run exactly ONCE, not every
        epoch. The register_* calls do self-guard with *_locked flags,
        but relying on that is how the old Phase 2 block ended up
        calling register_pde_scale twice in a row.
        """
        device = train_data.device

        if phase == 2:
            # Restore the best-generalizing Phase 1 surface BEFORE
            # anything reads its derivatives: pde_scale, mass_scale and
            # the collocation cache are all computed from this surface
            # and then LOCKED, so restoring afterwards would be too late.
            if not self._phase1_restored:
                best_path = self.save_name + '_best_phase1_model' if self.save_name else None
                if best_path is not None and os.path.exists(best_path):
                    weights = torch.load(best_path, map_location=device)
                    self.model.load_state_dict(weights)
                    print(f"Restored best Phase-1 surface from epoch {best_phase1_idx} "
                          f"(val GLS = {best_phase1_val:.4e})", flush=True)
                self._phase1_restored = True

            self.model.register_pde_scale(train_data)
            self.model.register_mass_scale(train_data)
            if not hasattr(self.model, '_collocation_cache'):
                self.model.refresh_collocation_cache()

        elif phase == 3:
            if not getattr(self.model, 'l0_scale_locked', False):
                # Start Phase 3 from the best reaction Phase 2 found, not
                # from wherever it happened to oscillate to. Must precede
                # register_l0_scale so the floor is measured at that state.
                self.model.restore_best_reaction()
                self.model.register_l0_scale()

    # ==================================================================
    # TRAINING
    # ==================================================================
    def fit(self, train_data, val_data, epochs, batch_size, l0_weight=0,
            initial_epoch=0, early_stopping=None, best_train_loss=None,
            best_val_loss=None, lr_dec_epoch=None, lr_dec_prop=1.0, rel_save_thresh=0.0,
            phase1_early_stopping=None, phase_ends=None):

        start_time = time.time()

        if initial_epoch == 0:
            self.param_history = {
                'raw_w_unscaled': [],
                'effective_unscaled': [],
                'diffusion_coeffs': [],
                'epoch': []
            }
            best_train_loss = 1e12 if best_train_loss is None else best_train_loss
            best_val_loss = 1e12 if best_val_loss is None else best_val_loss
            best_train_idx = 0
            best_val_idx = 0
            last_improved = 0
        else:
            # Recover indices when resuming from a checkpoint!
            if self.train_loss_dict['loss']:
                best_train_loss = min(self.train_loss_dict['loss'])
                best_train_idx = self.train_loss_dict['loss'].index(best_train_loss)
            else:
                best_train_loss = 1e12
                best_train_idx = initial_epoch

            if self.val_loss_dict['loss']:
                best_val_loss = min(self.val_loss_dict['loss'])
                best_val_idx = self.val_loss_dict['loss'].index(best_val_loss)
                last_improved = best_val_idx
            else:
                best_val_loss = 1e12
                best_val_idx = initial_epoch
                last_improved = initial_epoch

        if phase_ends is None:
            phase_ends = (int(0.05 * epochs), int(0.2 * epochs), int(0.3 * epochs))

        phase_1_end, phase_2_end, phase_3_end = phase_ends

        if not 0 < phase_1_end < phase_2_end < phase_3_end <= epochs:
            raise ValueError(
                f"phase_ends must be strictly increasing and within epochs: "
                f"got {phase_ends} with epochs={epochs}")

        if self.scheduler is not None:
            sched_steps = getattr(self.scheduler, 'total_steps', None)
            if sched_steps is not None and sched_steps != phase_1_end:
                print(f"WARNING: scheduler total_steps={sched_steps} != phase_1_end="
                      f"{phase_1_end}. The scheduler only steps during Phase 1, so "
                      f"Phase 1 will end mid-ramp (or the ramp will finish early).",
                      flush=True)

        last_improved = initial_epoch
        # --- Phase 1 surface-fitter tracking ---
        # Phase 1 originally had NO best-model tracking (the
        # best_val/save_best_val/early_stopping logic was all gated
        # behind `if phase == 4`), so Phase 2 inherited whatever surface
        # epoch phase_1_end happened to land on -- overfit or not. That
        # surface's derivatives then feed register_pde_scale and the
        # collocation cache, so an overfit surface propagates into every
        # later phase.
        prior_phase1 = self.val_loss_dict['gls'][:phase_1_end]
        if prior_phase1:
            best_phase1_val = min(prior_phase1)
            best_phase1_idx = prior_phase1.index(best_phase1_val)
        else:
            best_phase1_val = 1e12
            best_phase1_idx = initial_epoch
        phase1_last_improved = best_phase1_idx
        # Guard against a resumed Phase 2+ run clobbering its restored
        # model with the Phase 1 surface.
        self._phase1_restored = (initial_epoch >= phase_1_end)
        # Resumed runs shouldn't re-fire phase-entry work for a phase
        # they're already in the middle of; _on_phase_entry's internal
        # guards handle the register_* calls, and this keeps the banner
        # honest.
        self._last_phase = None

        for epoch in range(initial_epoch, epochs):
            # -----------------------------
            # 1. Determine Phase
            # -----------------------------
            if epoch < phase_1_end:
                phase = 1
            elif epoch < phase_2_end:
                phase = 2
            elif epoch < phase_3_end:
                phase = 3
            else:
                phase = 4

            # -----------------------------
            # 2. Apply phase settings
            # -----------------------------
            phase_bounds = (phase_1_end, phase_2_end, phase_3_end)
            entering_new_phase = (phase != self._last_phase)

            if entering_new_phase:
                self._on_phase_entry(phase, epoch, train_data,
                                     best_phase1_idx, best_phase1_val)
                self._last_phase = phase

            base_weights = self._apply_phase(phase, epoch, l0_weight,
                                             phase_bounds, train_data.device)
            mass_active = self.PHASE_CONFIG[phase]['mass_active']

            # Print AFTER _apply_phase so the banner shows the LRs actually
            # in effect, not the ones left over from the previous phase.
            if entering_new_phase:
                lrs = {pg.get('name'): pg['lr'] for pg in self.optimizer.param_groups}
                print(f"=== Phase {phase} @ epoch {epoch} | lrs: "
                      + ", ".join(f"{k}={v:.2e}" for k, v in lrs.items()) + " ===\n", flush=True)

            # Sample the full-cache PDE loss periodically through Phase 2 so
            # register_l0_scale can use the BEST floor rather than whatever
            # the oscillation left behind at Phase 3 entry. The reaction
            # weights swing by more than the explainable gap itself, so an
            # instantaneous reading is decided by trough-vs-peak luck.
            # Cheap: a reaction forward on cached points, no derivatives.
            if phase == 2 and epoch % 100 == 0:
                self.model._current_epoch = epoch
                self.model.track_best_pde()

            # --- The Clean Slate ---
            # Wipe the memory of Phase 1-3 losses the moment the
            # Regularization Phase (Phase 4) begins.
            if epoch == phase_3_end:
                best_train_loss = 1e12
                best_val_loss = 1e12
                best_train_idx = epoch  # Protects against UnboundLocalError
                best_val_idx = epoch    # Protects against UnboundLocalError
                last_improved = epoch

            # -----------------------------
            # 3. Train Step
            # -----------------------------
            self.train = True
            self.val = False

            self.model.train()
            epoch_start_time = time.time()

            if epoch % 1000 == 0:
                fn = f'{self.dir_name}/equation.txt'
                with open(fn, 'a') as file:
                    file.write(f'Equation Epoch {epoch}\n')
                    for line in self.model.equations_as_strings():
                        file.write(f'{line}\n')
                    file.write(f'\n')

                    if not self.model.diff_coeffs:
                        file.write(f'{[D.item() for D in self.model.diffusion_fitter()]}\n\n')

                param_snapshot = self.model.extract_params(full=False)  # list, one dict per equation
                self.param_history['epoch'].append(epoch)
                self.param_history['raw_w_unscaled'].append([p['raw_w_unscaled'] for p in param_snapshot])
                self.param_history['effective_unscaled'].append([p['effective_unscaled'] for p in param_snapshot])

                if not self.model.diff_coeffs:
                    self.param_history['diffusion_coeffs'].append([D.item() for D in self.model.diffusion_fitter()])

            train_losses = 0
            train_gls_losses = 0
            train_pde_losses = 0
            train_reg_losses = 0
            train_mass_losses = 0

            perm = torch.randperm(train_data.size(0))

            for i in range(0, len(train_data), batch_size):
                idx = perm[i:i+batch_size]
                x_true = train_data[idx, :-self.species].detach().clone().requires_grad_(True)
                y_true = train_data[idx, -self.species:].detach().clone().requires_grad_(True)

                self.optimizer.zero_grad()

                y_pred = self.model(x_true)

                raw_gls, raw_pde, raw_l0, raw_softwall, raw_mass = self.loss(y_pred, y_true, epoch, phase)

                weighted_losses = torch.stack([raw_gls, raw_pde, raw_l0]) * base_weights

                mass_term = self.mass_weight * raw_mass if mass_active else torch.tensor(0.0, device=x_true.device)

                train_loss = torch.sum(weighted_losses) + raw_softwall + mass_term

                train_gls_loss = weighted_losses[0]
                train_pde_loss = weighted_losses[1]
                train_reg_loss = weighted_losses[2] + raw_softwall

                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses += train_loss.item() * len(x_true)
                train_gls_losses += train_gls_loss.item() * len(x_true)
                train_pde_losses += train_pde_loss.item() * len(x_true)
                train_reg_losses += train_reg_loss.item() * len(x_true)
                train_mass_losses += mass_term.item() * len(x_true)

            if self.scheduler is not None and self.PHASE_CONFIG[phase]['scheduler']:
                self.scheduler.step()

            self.train_loss_dict['loss'].append(np.sum(train_losses) / len(train_data))
            self.train_loss_dict['gls'].append(np.sum(train_gls_losses) / len(train_data))
            self.train_loss_dict['pde'].append(np.sum(train_pde_losses) / len(train_data))
            self.train_loss_dict['reg'].append(np.sum(train_reg_losses) / len(train_data))
            self.train_loss_dict['mass'].append(np.sum(train_mass_losses) / len(train_data))

            if phase == 4:
                rel_diff = (best_train_loss - self.train_loss_dict['loss'][-1]) / best_train_loss
                if rel_diff > rel_save_thresh:
                    best_train_loss = self.train_loss_dict['loss'][-1]
                    best_train_idx = epoch
                    if self.save_best_train:
                        self.save(self.save_name + '_best_train')

            # -----------------------------
            # 4. Validation Step
            # -----------------------------
            self.train = False
            self.val = True

            self.model.eval()

            val_losses = 0
            val_gls_losses = 0
            val_pde_losses = 0
            val_reg_losses = 0
            val_mass_losses = 0

            no_perm = torch.arange(val_data.size(0))

            for i in range(0, len(val_data), batch_size):
                idx = no_perm[i:i+batch_size]
                x_true = val_data[idx, :-self.species].detach().clone().requires_grad_(True)
                y_true = val_data[idx, -self.species:].detach().clone().requires_grad_(True)

                self.optimizer.zero_grad()

                y_pred = self.model(x_true)

                raw_gls, raw_pde, raw_l0, raw_softwall, raw_mass = self.loss(y_pred, y_true, epoch, phase)

                weighted_losses = torch.stack([raw_gls, raw_pde, raw_l0]) * base_weights

                mass_term = self.mass_weight * raw_mass if mass_active else torch.tensor(0.0, device=x_true.device)

                val_loss = torch.sum(weighted_losses) + raw_softwall + mass_term

                val_gls_loss = weighted_losses[0]
                val_pde_loss = weighted_losses[1]
                val_reg_loss = weighted_losses[2] + raw_softwall

                val_losses += val_loss.item() * len(x_true)
                val_gls_losses += val_gls_loss.item() * len(x_true)
                val_pde_losses += val_pde_loss.item() * len(x_true)
                val_reg_losses += val_reg_loss.item() * len(x_true)
                val_mass_losses += mass_term.item() * len(x_true)

            self.val_loss_dict['loss'].append(np.sum(val_losses) / len(val_data))
            self.val_loss_dict['gls'].append(np.sum(val_gls_losses) / len(val_data))
            self.val_loss_dict['pde'].append(np.sum(val_pde_losses) / len(val_data))
            self.val_loss_dict['reg'].append(np.sum(val_reg_losses) / len(val_data))
            self.val_loss_dict['mass'].append(np.sum(val_mass_losses) / len(val_data))

            if phase == 1:
                # Use GLS, not total loss: in Phase 1 the reaction and
                # diffusion params are frozen, so total loss is just GLS
                # plus a constant soft-wall offset. GLS alone measures
                # the surface fitter's generalization, which is the only
                # thing being trained here.
                current_gls = self.val_loss_dict['gls'][-1]
                rel_diff = (best_phase1_val - current_gls) / best_phase1_val
                if rel_diff > rel_save_thresh:
                    best_phase1_val = current_gls
                    best_phase1_idx = epoch
                    phase1_last_improved = epoch
                    if self.save_name is not None:
                        self.save(self.save_name + '_best_phase1')

                # Don't let early stopping fire while OneCycleLR is still
                # annealing -- its val loss characteristically rises at
                # peak LR then drops during the anneal, so patience-based
                # stopping tends to trigger right before the improvement
                # it's waiting for.
                scheduler_done = (self.scheduler is None
                                  or epoch >= int(0.9 * getattr(self.scheduler, 'total_steps', 0)))

                if (phase1_early_stopping is not None and scheduler_done
                        and epoch - phase1_last_improved >= phase1_early_stopping):
                    print(f"Phase 1 early stopping at epoch {epoch} "
                          f"(no val GLS improvement in {phase1_early_stopping} epochs)",
                          flush=True)
                    # Don't break -- just end Phase 1 here so the next
                    # epoch falls through to Phase 2. Phases 2-4 keep
                    # their original end epochs and simply get longer.
                    phase_1_end = epoch + 1

            if phase == 4:
                rel_diff = (best_val_loss - self.val_loss_dict['loss'][-1]) / best_val_loss
                if rel_diff > rel_save_thresh:
                    best_val_loss = self.val_loss_dict['loss'][-1]
                    best_val_idx = epoch
                    if self.save_best_val:
                        self.save(self.save_name + '_best_val')

                    last_improved = epoch

                if early_stopping is not None and epoch - last_improved >= early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            elapsed, remaining, ms = time_remaining(
                current_iter=epoch+1,
                total_iter=initial_epoch+epochs,
                start_time=start_time,
                previous_time=epoch_start_time,
                ops_per_iter=batch_size)

            if epoch % 1000 == 0:
                p = 'Epoch {0}'.format(epoch)
                p += ' | Train loss = {0:1.4e}'.format(self.train_loss_dict['loss'][-1])
                p += ' | Val loss = {0:1.4e}'.format(self.val_loss_dict['loss'][-1])
                p += ' | Val GLS = {0:1.4e},'.format(self.val_loss_dict['gls'][-1])
                p += ' Val PDE = {0:1.4e},'.format(self.val_loss_dict['pde'][-1])
                p += ' Val Reg = {0:1.4e}'.format(self.val_loss_dict['reg'][-1])
                p += ' Val Mass = {0:1.4e}'.format(self.val_loss_dict['mass'][-1])
                if not self.model.diff_coeffs:
                    D = self.model.diffusion_fitter()
                    names = self.model.species_names
                    p += ' | ' + ', '.join(f'D{names[i]}={D[i].item():.4e}' for i in range(len(D)))
                p += ' | Remaining = ' + remaining + '           '
                print(p, flush=True)

            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    # Decay the BASE lrs, not param_groups directly --
                    # _apply_phase overwrites param_groups from base_lrs
                    # every epoch, so decaying param_groups alone would be
                    # silently undone on the very next iteration.
                    for name in self.base_lrs:
                        self.base_lrs[name] *= lr_dec_prop

            # -----------------------------
            # 5. Checkpoint Saving
            # -----------------------------
            if epoch % 50 == 0 or epoch == epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                    'train_loss_dict': self.train_loss_dict,
                    'val_loss_dict': self.val_loss_dict,
                    'param_history': self.param_history
                }
                torch.save(checkpoint, f'{self.dir_name}/latest_checkpoint.pt')

        elapsed, remaining, ms = time_remaining(
            current_iter=epoch+1,
            total_iter=initial_epoch+epochs,
            start_time=start_time,
            previous_time=epoch_start_time,
            ops_per_iter=batch_size)

        if self.save_best_val:
            best_idx = best_val_idx
        elif self.save_best_train:
            best_idx = best_train_idx
        else:
            best_idx = -1

        p = 'Epoch {0}'.format(epoch)
        p += ' | Train loss = {0:1.4e}'.format(self.train_loss_dict['loss'][best_idx])
        p += ' | Val loss = {0:1.4e}'.format(self.val_loss_dict['loss'][best_idx])
        p += ' | Val GLS = {0:1.4e},'.format(self.val_loss_dict['gls'][best_idx])
        p += ' Val PDE = {0:1.4e},'.format(self.val_loss_dict['pde'][best_idx])
        p += ' Val Reg = {0:1.4e}'.format(self.val_loss_dict['reg'][best_idx])

        p += ' | Elapsed = ' + elapsed + '           '
        print(p, flush=True)

        return self.param_history, self.train_loss_dict, self.val_loss_dict

    # ==================================================================
    # CHECKPOINT / IO
    # ==================================================================
    def load_checkpoint(self, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        resume_epoch = checkpoint['epoch'] + 1

        if self.scheduler is not None and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.scheduler.last_epoch = resume_epoch

        self.train_loss_dict = checkpoint['train_loss_dict']
        self.val_loss_dict = checkpoint['val_loss_dict']
        self.param_history = checkpoint['param_history']

        return resume_epoch

    def predict(self, inputs):
        self.model.eval()
        return self.model(inputs)

    def save(self, save_name):
        torch.save(self.model.state_dict(), save_name+'_model')
        if self.save_opt and self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), save_name+'_opt')
        if self.save_reg and self.regularizer is not None:
            torch.save(self.regularizer.state_dict(), save_name+'_reg')

    def load(self, model_weights, opt_weights=None, reg_weights=None, device=None):
        weights = torch.load(model_weights, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()

        if opt_weights is not None:
            params = torch.load(opt_weights, map_location=device)
            self.optimizer.load_state_dict(params)

        if reg_weights is not None:
            params = torch.load(reg_weights, map_location=device)
            self.regularizer.load_state_dict(params)

    def load_best_train(self, device=None):
        name = self.save_name+'_best_train_model'
        weights = torch.load(name, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()

        if self.save_opt and self.optimizer is not None:
            name = self.save_name+'_best_train_opt'
            params = torch.load(name, map_location=device)
            self.optimizer.load_state_dict(params)

        if self.save_reg and self.regularizer is not None:
            name = self.save_name+'_best_train_reg'
            params = torch.load(name, map_location=device)
            self.regularizer.load_state_dict(params)

    def load_best_val(self, device=None):
        name = self.save_name+'_best_val_model'
        weights = torch.load(name, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()

        if self.save_opt and self.optimizer is not None:
            name = self.save_name+'_best_val_opt'
            params = torch.load(name, map_location=device)
            self.optimizer.load_state_dict(params)

        if self.save_reg and self.regularizer is not None:
            name = self.save_name+'_best_val_reg'
            params = torch.load(name, map_location=device)
            self.regularizer.load_state_dict(params)