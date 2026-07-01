import torch, time
import numpy as np

from modules.utils.time_remaining import *
from modules.binn_eql.gated_relobralo import GatedReLoBRaLo
from modules.utils.gradient import gradient

# --- ENFORCE TRUE FP32 PRECISION ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class model_wrapper():
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
                 save_reg=False):
        
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
        self.train_loss_dict = {'loss': [], 'gls': [], 'pde': [], 'reg': []}
        self.val_loss_dict = {'loss': [], 'gls': [], 'pde': [], 'reg': []}
        self.train = False
        self.val = False
        
        if self.save_name is None:
            self.save_best_train = False
            self.save_best_val = False
            self.save_opt = False
            self.save_reg = False
        
    def fit(self, train_data, val_data, epochs, batch_size, l0_weight=0,
            initial_epoch=0, early_stopping=None, best_train_loss=None,
            best_val_loss=None, lr_dec_epoch=None, lr_dec_prop=1.0, rel_save_thresh=0.0):
                
        start_time = time.time()
        
        # Protect histories from being overwritten if we are resuming
        if initial_epoch == 0:
            self.param_history = {
                'raw_w_unscaled': [], 
                'effective_unscaled': [],
                'diffusion_coeffs': [], 
                'epoch': []
            }
            best_train_loss = 1e12 if best_train_loss is None else best_train_loss
            best_val_loss = 1e12 if best_val_loss is None else best_val_loss  
        else:
            # Recover the best losses from the loaded history
            best_train_loss = min(self.train_loss_dict['loss']) if self.train_loss_dict['loss'] else 1e12
            best_val_loss = min(self.val_loss_dict['loss']) if self.val_loss_dict['loss'] else 1e12
        
        self.relobralo = GatedReLoBRaLo(num_losses=3, temperature=0.1, alpha=0.99, device=train_data.device)
        
        phase_1_end = int(0.2 * epochs)
        phase_2_end = int(0.4 * epochs)
        phase_3_end = int(0.6 * epochs)

        # Initialize early stopping tracker before the loop starts
        last_improved = initial_epoch
        
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
            # 2. Determine Weights and learning rates
            # -----------------------------
            if phase == 1:
                # Phase 1: Data Only
                base_weights = torch.tensor([1.0, 0.0, 0.0], device=train_data.device)

                # Let scheduler handle LR, add decay for smooth surface
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['weight_decay'] = 1e-5

                for p in self.model.surface_fitter.parameters(): p.requires_grad = True
                for p in self.model.reaction.parameters(): p.requires_grad = False
                if self.model.diffusion_fitter:
                    for p in self.model.diffusion_fitter.parameters(): p.requires_grad = False
                  
            elif phase == 2:
                # Phase 2: Physics On, No Reg
                base_weights = torch.tensor([0.0, 1.0, 0.0], device=train_data.device)      
                          
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['lr'] = 0.0
                    elif pg.get('name') == 'reaction':
                        pg['lr'] = 1e-3
                    elif pg.get('name') == 'diffusion':
                        # pg['lr'] = 0
                        pg['lr'] = 1e-3

                for p in self.model.surface_fitter.parameters(): p.requires_grad = False
                for p in self.model.reaction.parameters(): p.requires_grad = True
                if self.model.diffusion_fitter:
                    for p in self.model.diffusion_fitter.parameters(): p.requires_grad = True

                # ---------------------------------------------------------
                # ONE-TIME PDE NORMALIZATION (Robust Top 5% Scaling)
                # ---------------------------------------------------------
                if not hasattr(self.model, 'pde_scales_locked'):
                    print("\n--- Phase 2 Start: Calculating Robust PDE Scales (Top 1%) ---")
                    
                    self.model.surface_fitter.eval() 
                    
                    all_u_sq = []
                    all_v_sq = []
                    total_points = len(train_data)
                    chunk_size = 50_000 
                    
                    for chunk_start in range(0, total_points, chunk_size):
                        chunk = train_data[chunk_start:chunk_start+chunk_size].clone().requires_grad_(True)
                        outputs = self.model.surface_fitter(self.model.normalize(chunk[:, :self.model.dimensions+1]))
                        
                        u_t = gradient(outputs[:, 0], chunk, order=1)[:, self.model.dimensions] 
                        v_t = gradient(outputs[:, 1], chunk, order=1)[:, self.model.dimensions]
                        
                        all_u_sq.append((u_t**2).detach().cpu())
                        all_v_sq.append((v_t**2).detach().cpu())
                        
                        del chunk, outputs, u_t, v_t
                        
                    self.model.surface_fitter.train() 
                    
                    global_u_sq = torch.cat(all_u_sq)
                    global_v_sq = torch.cat(all_v_sq)
                    
                    k_percent = 0.01
                    k_points = int(k_percent * total_points)
                    
                    robust_max_u = torch.topk(global_u_sq, k_points).values.mean().item()
                    robust_max_v = torch.topk(global_v_sq, k_points).values.mean().item()
                    # robust_max_u = torch.quantile(global_u_sq, 0.90).item()
                    # robust_max_v = torch.quantile(global_v_sq, 0.90).item()    

                    # <-- FIXED: Attach scales directly to the BINN model -->
                    self.model.pde_scale_u = robust_max_u + 1e-6
                    self.model.pde_scale_v = robust_max_v + 1e-6
                    self.model.pde_scales_locked = True 
                    
                    print(f"Locked Top {k_percent*100}% Scales -> u: {self.model.pde_scale_u:.4e}, v: {self.model.pde_scale_v:.4e}\n")

            elif phase == 3:
                # Phase 3: Physics On, Ramp Reg
                phase_duration = phase_3_end - phase_2_end
                progress = ((epoch - phase_2_end) / phase_duration)
                base_weights = torch.tensor([0.0, 1.0, l0_weight*progress], device=train_data.device)    
                            
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['lr'] = 0.0
                    elif pg.get('name') == 'reaction':
                        pg['lr'] = 1e-3
                    elif pg.get('name') == 'diffusion':
                        pg['lr'] = 1e-3
                
                for p in self.model.surface_fitter.parameters(): p.requires_grad = False
                for p in self.model.reaction.parameters(): p.requires_grad = True
                if self.model.diffusion_fitter:
                    for p in self.model.diffusion_fitter.parameters(): p.requires_grad = True

            elif phase == 4:
                # Phase 4: Max Reg
                base_weights = torch.tensor([0.0, 1.0, l0_weight], device=train_data.device)  
                              
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['lr'] = 0.0
                    elif pg.get('name') == 'reaction':
                        pg['lr'] = 1e-3
                    elif pg.get('name') == 'diffusion':
                        pg['lr'] = 1e-3

                for p in self.model.surface_fitter.parameters(): p.requires_grad = False
                for p in self.model.reaction.parameters(): p.requires_grad = True
                if self.model.diffusion_fitter:
                    for p in self.model.diffusion_fitter.parameters(): p.requires_grad = True

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
                    for term in self.model.generate_equation():
                        file.write(f'{term}\n')
                    file.write(f'\n')
                    
                    if not self.model.diff_coeffs:          
                        file.write(f'{[D.item() for D in self.model.diffusion_fitter()]}\n\n')

                param_snapshot = self.model.extract_params(full=False)
                self.param_history['epoch'].append(epoch)
                self.param_history['raw_w_unscaled'].append(param_snapshot['raw_w_unscaled'])
                self.param_history['effective_unscaled'].append(param_snapshot['effective_unscaled'])

                if not self.model.diff_coeffs:
                    self.param_history['diffusion_coeffs'].append([D.item() for D in self.model.diffusion_fitter()])
                else:
                    # Fallback just in case you run a fixed-diffusion experiment
                    self.param_history['diffusion_coeffs'].append(self.model.diff_coeffs)
                
            train_losses = 0
            train_gls_losses = 0
            train_pde_losses = 0
            train_reg_losses = 0

            perm = torch.randperm(train_data.size(0))
            
            for i in range(0, len(train_data), batch_size):
                idx = perm[i:i+batch_size]
                x_true = train_data[idx, :-self.species].detach().clone().requires_grad_(True)
                y_true = train_data[idx, -self.species:].detach().clone().requires_grad_(True)
                
                self.optimizer.zero_grad()
                                    
                y_pred = self.model(x_true)
                                    
                # --- RELOBRALO INTEGRATION ---                   
                # 1. Capture the 4 RAW unweighted tensors
                raw_gls, raw_pde, raw_l0, raw_softwall = self.loss(y_pred, y_true, epoch)
                
                # 2. Stack only the 3 competing losses for ReLoBRaLo
                raw_relo_losses = torch.stack([raw_gls, raw_pde, raw_l0])
                
                # 3. Compute dynamic lambdas (In Val step, use current_lambdas instead)
                # dynamic_lambdas = self.relobralo.compute_weights(raw_relo_losses, base_weights)
                dynamic_lambdas = 1
                
                # 4. Multiply and sum the balanced losses
                weighted_losses = raw_relo_losses * base_weights * dynamic_lambdas
                
                # 5. Add the Soft Wall strictly ON TOP (Bypasses all weights and phases)
                train_loss = torch.sum(weighted_losses) + raw_softwall
                
                # Extract weighted components for logging (Adding softwall to Reg for display)
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

            if self.scheduler is not None and phase == 1:
                self.scheduler.step()
                
            self.train_loss_dict['loss'].append(np.sum(train_losses) / len(train_data))
            self.train_loss_dict['gls'].append(np.sum(train_gls_losses) / len(train_data))
            self.train_loss_dict['pde'].append(np.sum(train_pde_losses) / len(train_data))
            self.train_loss_dict['reg'].append(np.sum(train_reg_losses) / len(train_data))
                                
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
            
            no_perm  = torch.arange(val_data.size(0))
            
            for i in range(0, len(val_data), batch_size):
                idx = no_perm[i:i+batch_size]
                x_true = val_data[idx, :-self.species].detach().clone().requires_grad_(True)                        
                y_true = val_data[idx, -self.species:].detach().clone().requires_grad_(True)    
                  
                self.optimizer.zero_grad()
                                                
                # 1. Forward Pass
                y_pred = self.model(x_true)
                
                # --- RELOBRALO VALIDATION ---                   
                # 2. Capture the 4 RAW unweighted tensors
                raw_gls, raw_pde, raw_l0, raw_softwall = self.loss(y_pred, y_true, epoch)
                
                # 3. Stack only the 3 competing losses
                raw_relo_losses = torch.stack([raw_gls, raw_pde, raw_l0])
                
                # 4. Read the current lambdas (DO NOT call compute_weights here!)
                # current_lambdas = self.relobralo.lambdas.to(x_true.device)
                current_lambdas = 1
                
                # 5. Multiply and sum the balanced losses
                weighted_losses = raw_relo_losses * base_weights * current_lambdas
                
                # 6. Add the Soft Wall strictly ON TOP
                val_loss = torch.sum(weighted_losses) + raw_softwall
                
                # 7. Extract weighted components for logging
                val_gls_loss = weighted_losses[0]
                val_pde_loss = weighted_losses[1]
                val_reg_loss = weighted_losses[2] + raw_softwall    
                                
                val_losses += val_loss.item() * len(x_true)
                val_gls_losses += val_gls_loss.item() * len(x_true)
                val_pde_losses += val_pde_loss.item() * len(x_true)
                val_reg_losses += val_reg_loss.item() * len(x_true)

            self.val_loss_dict['loss'].append(np.sum(val_losses) / len(val_data))
            self.val_loss_dict['gls'].append(np.sum(val_gls_losses) / len(val_data))
            self.val_loss_dict['pde'].append(np.sum(val_pde_losses) / len(val_data))
            self.val_loss_dict['reg'].append(np.sum(val_reg_losses) / len(val_data))

            if phase == 4:
                rel_diff = (best_val_loss - self.val_loss_dict['loss'][-1]) / best_val_loss
                if rel_diff > rel_save_thresh:
                    best_val_loss = self.val_loss_dict['loss'][-1]
                    best_val_idx = epoch
                    if self.save_best_val:
                        self.save(self.save_name + '_best_val')
                    
                    # Reset Early Stopping Counter
                    last_improved = epoch

                # Trigger Early Stopping
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
                p += ' | Remaining = ' + remaining + '           '
                print(p, flush=True)
                                            
            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= lr_dec_prop

            # -----------------------------
            # 5. Checkpoint Saving (ADD THIS TO THE VERY END OF THE EPOCH LOOP)
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
    
    def load_checkpoint(self, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # --- THE FIX ---
        # Get the true epoch from the checkpoint dictionary
        resume_epoch = checkpoint['epoch'] + 1
        
        if self.scheduler is not None and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            # Manually sync the scheduler's internal clock to the true resume epoch
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