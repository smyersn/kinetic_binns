import torch, time, sys, pdb, os
import numpy as np
import psutil

from modules.utils.time_remaining import *

class model_wrapper():
   
    '''
    Utility function that wraps around a PyTorch model. It allows for easy
    training and saving/loading feed-forward models. Work in progress.
   
    Args:
        model        (callable): Model.
        optimizer    (callable): Optimizer.
        loss         (callable): Loss function that inputs (pred, true).
        regularizer  (callable): Regularization that inputs (model, inputs, 
                                  true, pred).
        scheduler    (callable): Learning rate scheduler.
        save_name      (string): Model name for saving model/opt weights.
        save_best_train  (bool): Indicator for saving on best train loss.
        save_best_val    (bool): Indicator for saving on best val loss.
        save_opt         (bool): Indicator for saving optimizer weights.
   
    Inputs:
        x       (tensor/generator): Input data.
        y       (tensor/generator): Target data.
        batch_size           (int): Batch size.
        epochs               (int): Total number of epochs.
        validation_data     (list): Input and target validation data.
        shuffle             (bool): Whether to shuffle data each epoch.
        class_weight        (list): NOT IMPLEMENTED
        sample_weight       (list): NOT IMPLEMENTED
        initial_epoch        (int): Initial epoch to start training.
        validation_freq      (int): NOT IMPLEMENTED
        early_stopping       (int): Number of epochs since validation improved.
        best_train_loss    (float): Best loss on training set.
        best_val_loss      (float): Best loss on validation set.
        include_val_reg     (bool): Inclusion of regularizer in val loss.
        lr_dec_epoch         (int): Decrease lr after this many epochs.
        lr_dec_prop        (float): Value <= 1 to multiply learning rate.
        rel_save_thresh    (float): Rel. diff. btwn losses before saving.
        
    Returns:
        self.train_loss_dict (dict): Training errors per epoch.
        val_loss_dict       (dict): Validation errors per epoch.
    '''
   
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
        
        # if no name specified, don't save weights
        if self.save_name is None:
            self.save_best_train = False
            self.save_best_val = False
            self.save_opt = False
            self.save_reg = False
        
    def fit(self,
            train_data,
            val_data,
            pde_weight,
            l0_weight,
            warm_up,
            lux_tax,
            batch_size=None,
            epochs=1,
            initial_epoch=0,
            early_stopping=None,
            best_train_loss=None,
            best_val_loss=None,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_save_thresh=0.0):
                
        # initialize book keeping
        start_time = time.time()
        best_train_loss = 1e12 if best_train_loss is None else best_train_loss
        best_val_loss = 1e12 if best_val_loss is None else best_val_loss  
        
        # simple history container
        self.param_history = {'raw_w_unscaled': [], 'effective_unscaled': [],'epoch': []}
        
        phase_1_end = 20_000
        phase_2_end = 20_000 + int(warm_up * 0.5)
        phase_3_end = 20_000 + int(warm_up * 1)
        last_improved = 20_000 + int(warm_up * 1)
              
        # loop over epochs
        for epoch in range(initial_epoch, initial_epoch + epochs):
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
                
            # Apply Freezing (Idempotent, safe to call every epoch)
            self.set_training_phase(phase)
            
            # -----------------------------
            # 2. Determine Weights and learning rates
            # -----------------------------
            
            # Phase 1: Data Only (The Sprint)
            if phase == 1:
                gls_weight_eff = 1.0
                pde_weight_eff = 0.0
                l0_weight_eff = 0.0
                # LR Strategy: Trust the OneCycleLR Scheduler completely.
                
            # Phase 2: Physics On, No Reg
            elif phase == 2:
                gls_weight_eff = 0.0
                pde_weight_eff = pde_weight
                l0_weight_eff = 0.0
                
                # LR Strategy: Manual Constant (Stabilize Surface, Wake Reaction)
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['lr'] = 0.0   # Lock it down (Micro-adjustments only)
                    elif pg.get('name') == 'reaction':
                        pg['lr'] = 1e-3   # Wake up! (Standard learning)
                    elif pg.get('name') == 'diffusion':
                        pg['lr'] = 1e-3   # Wake up!

            # Phase 3: Physics On, Ramp Reg (The Selection)
            elif phase == 3:
                gls_weight_eff = 0.0
                pde_weight_eff = pde_weight
                
                # Calculate progress through Phase 3 (0.0 to 1.0)
                phase_duration = phase_3_end - phase_2_end
                progress = (epoch - phase_2_end) / phase_duration
                l0_weight_eff = progress * l0_weight
                
                # LR Strategy: Constant (Keep steady pressure against L0 tax)
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['lr'] = 0.0   # Keep locked
                    elif pg.get('name') == 'reaction':
                        pg['lr'] = 1e-3   # Keep strong to fight Regularization
                    elif pg.get('name') == 'diffusion':
                        pg['lr'] = 1e-3

            # Phase 4: Max Reg (The Alignment / Fine Tuning)
            elif phase == 4:
                gls_weight_eff = 0.0
                pde_weight_eff = pde_weight
                l0_weight_eff = l0_weight
                
                # LR Strategy: Decay Reaction for precision
                for pg in self.optimizer.param_groups:
                    if pg.get('name') == 'surface':
                        pg['lr'] = 0.0   # Still locked
                    elif pg.get('name') == 'reaction':
                        # pg['lr'] = 1e-4   # Drop for fine-tuning
                        pg['lr'] = 1e-3   # Drop for fine-tuning
                    elif pg.get('name') == 'diffusion':
                        # pg['lr'] = 1e-4
                        pg['lr'] = 1e-3   # Drop for fine-tuning
                        
            # -----------------------------
            # 3. Train Step
            # -----------------------------
            self.train = True
            self.val = False
                    
            self.model.train()
            epoch_start_time = time.time()
            
            # Print equation every 1000 epochs                                                                        
            if epoch % 1000 == 0:
                fn = f'{self.dir_name}/equation.txt'
                file = open(fn, 'a')
                
                file.write(f'Equation Epoch {epoch}\n')
                for term in self.model.generate_equation():
                    file.write(f'{term}\n')
                file.write(f'\n')

                file.close()
                
                # Save history
                param_snapshot = self.model.extract_params(full=False)
                self.param_history['epoch'].append(epoch)
                self.param_history['raw_w_unscaled'].append(param_snapshot['raw_w_unscaled'])
                self.param_history['effective_unscaled'].append(param_snapshot['effective_unscaled'])
                
            # Create lists for epoch training losses
            train_losses = 0
            train_gls_losses = 0
            train_pde_losses = 0
            train_reg_losses = 0

            # Shuffle training data
            perm = torch.randperm(train_data.size(0))
            
            # loop over training batches
            for i in range(0, len(train_data), batch_size):
                idx = perm[i:i+batch_size]
                # x_true = train_data[idx, :-self.species].data.clone()
                # y_true = train_data[idx, -self.species:].data.clone()
                x_true = train_data[idx, :-self.species].detach().clone().requires_grad_(True)
                y_true = train_data[idx, -self.species:].detach().clone().requires_grad_(True)
                # zero out gradients
                self.optimizer.zero_grad()
                                    
                # require gradients
                # x_true.requires_grad = True
                
                # run the model
                y_pred = self.model(x_true)
                                    
                # compute loss and optional regularization
                train_loss, train_gls_loss, train_pde_loss, train_reg_loss = self.loss(y_pred, 
                                                                                       y_true, 
                                                                                       epoch,
                                                                                       gls_weight_eff,
                                                                                       pde_weight_eff,
                                                                                       l0_weight_eff,
                                                                                       lux_tax)
                                                                            
                # compute backward pass and update weights
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)                       
                self.optimizer.step()
                                        
                # Update losses
                train_losses += train_loss.item() * len(x_true)
                train_gls_losses += train_gls_loss.item() * len(x_true)
                train_pde_losses += train_pde_loss.item() * len(x_true)
                train_reg_losses += train_reg_loss.item() * len(x_true)

            # Step scheduler after all training batches
            if self.scheduler is not None and phase == 1:
                self.scheduler.step()
                
            # update book keeping for this epoch
            self.train_loss_dict['loss'].append(np.sum(train_losses) / len(train_data))
            self.train_loss_dict['gls'].append(np.sum(train_gls_losses) / len(train_data))
            self.train_loss_dict['pde'].append(np.sum(train_pde_losses) / len(train_data))
            self.train_loss_dict['reg'].append(np.sum(train_reg_losses) / len(train_data))
            
            if phase == 4:
                rel_diff = (best_train_loss - self.train_loss_dict['loss'][-1]) / best_train_loss
                
                if rel_diff > rel_save_thresh:
                    best_train_loss = self.train_loss_dict['loss'][-1]
                    
                    if self.save_best_train:
                        self.save(self.save_name + '_best_train')
                        
            # -----------------------------
            # 3. Validation Step
            # -----------------------------
            self.train = False
            self.val = True
            
            self.model.eval()
            
            # Create lists for epoch training losses
            val_losses = 0
            val_gls_losses = 0
            val_pde_losses = 0
            val_reg_losses = 0
            
            # Don't shuffle val data
            no_perm  = torch.arange(val_data.size(0))
            
            # loop over validation batches
            for i in range(0, len(val_data), batch_size):
                idx = no_perm[i:i+batch_size]
                # x_true = val_data[idx, :-self.species].data.clone()
                # y_true = val_data[idx, -self.species:].data.clone()
                x_true = val_data[idx, :-self.species].detach().clone().requires_grad_(True)                         
                y_true = val_data[idx, -self.species:].detach().clone().requires_grad_(True)     
                  
                self.optimizer.zero_grad()
                                                
                # require gradients
                # x_true.requires_grad = True
                                
                # run the model
                y_pred = self.model(x_true)
                
                # comptue loss
                val_loss, val_gls_loss, val_pde_loss, val_reg_loss = self.loss(y_pred,
                                                                               y_true,
                                                                               epoch,
                                                                               gls_weight_eff,
                                                                               pde_weight_eff,
                                                                               l0_weight_eff,
                                                                               lux_tax)
                
                val_losses += val_loss.item() * len(x_true)
                val_gls_losses += val_gls_loss.item() * len(x_true)
                val_pde_losses += val_pde_loss.item() * len(x_true)
                val_reg_losses += val_reg_loss.item() * len(x_true)

            # update book keeping for this epoch
            self.val_loss_dict['loss'].append(np.sum(val_losses) / len(val_data))
            self.val_loss_dict['gls'].append(np.sum(val_gls_losses) / len(val_data))
            self.val_loss_dict['pde'].append(np.sum(val_pde_losses) / len(val_data))
            self.val_loss_dict['reg'].append(np.sum(val_reg_losses) / len(val_data))

            if phase == 4:
                rel_diff = (best_val_loss - self.val_loss_dict['loss'][-1]) / best_val_loss
                
                if rel_diff > rel_save_thresh:
                    best_val_loss = self.val_loss_dict['loss'][-1]
                    
                    if self.save_best_val:
                        self.save(self.save_name + '_best_val')
                    
                    # Reset Early Stopping Counter
                    last_improved = epoch

                # Trigger Early Stopping (usually only in Phase 4)
                if epoch - last_improved >= early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break               
                                                           
            # update user
            elapsed, remaining, ms = time_remaining(
                current_iter=epoch+1,
                total_iter=initial_epoch+epochs,
                start_time=start_time,
                previous_time=epoch_start_time,
                ops_per_iter=batch_size)
            
            # prints
            if epoch % 1000 == 0:
                p = 'Epoch {0}'.format(epoch)
                p += ' | Train loss = {0:1.4e}'.format(self.train_loss_dict['loss'][-1])
                p += ' | Val loss = {0:1.4e}'.format(self.val_loss_dict['loss'][-1])
                p += ' | Remaining = ' + remaining + '           '
                #sys.stdout.write(p)
                print(p, flush=True)
                                            
            # optional learning rate annealing
            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= lr_dec_prop
                        
        # self.save(self.save_name+'_best_val')

        # final print readout
        elapsed, remaining, ms = time_remaining(
            current_iter=epoch+1,
            total_iter=initial_epoch+epochs,
            start_time=start_time,
            previous_time=epoch_start_time,
            ops_per_iter=batch_size)
        
        # prints
        if self.save_best_val:
            idx = np.argmin(self.val_loss_dict['loss'])
        elif self.save_best_train:
            idx = np.argmin(self.self.train_loss_dict['loss'])
        else:
            idx = -1
        p = 'Epoch {0}'.format(epoch)
        p += ' | Train loss = {0:1.4e}'.format(self.train_loss_dict['loss'][idx])
        p += ' | Val loss = {0:1.4e}'.format(self.val_loss_dict['loss'][idx])
        p += ' | Elapsed = ' + elapsed + '           '
        #sys.stdout.write(p)
        print(p, flush=True)
            
        return self.param_history, self.train_loss_dict, self.val_loss_dict
    
    def freeze_pruned_params(self):
        fc_weight = self.model.reaction.eql_layer.fc.weight

        # Generate mask w/ zeros at positions of pruned parameters
        with torch.no_grad(): # Ensure this operation doesn't track gradients
            mask = (fc_weight.data != 0).float()

        # Clear optimizer state for pruned parameters
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param is fc_weight: # Only target the specific pruned layer
                    if param in self.optimizer.state:
                        state = self.optimizer.state[param]
                        if 'exp_avg' in state:
                            state['exp_avg'].mul_(mask)
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].mul_(mask)
                                     
    def set_training_phase(self, phase):
        # Phase 1: Surface Fitting Only
        if phase == 1:
            # Unfreeze Surface
            for p in self.model.surface_fitter.parameters(): p.requires_grad = True
            
            # Freeze Physics (Reaction + Diffusion)
            for p in self.model.reaction.parameters(): p.requires_grad = False
            if self.model.diffusion_fitter:
                for p in self.model.diffusion_fitter.parameters(): p.requires_grad = False
                
        # Phase 2 & 3: Equation Discovery & Pruning (Surface Frozen)
        # We keep the surface frozen so the equation learns against a "stable target"
        elif phase == 2 or phase == 3:
            # Freeze Surface (CRITICAL)
            for p in self.model.surface_fitter.parameters(): p.requires_grad = False
            
            # Unfreeze Physics
            for p in self.model.reaction.parameters(): p.requires_grad = True
            if self.model.diffusion_fitter:
                for p in self.model.diffusion_fitter.parameters(): p.requires_grad = True
                
        # Phase 4: Joint Fine-Tuning
        elif phase == 4:
            # Unfreeze Everything
            for p in self.model.surface_fitter.parameters(): p.requires_grad = True
            for p in self.model.reaction.parameters(): p.requires_grad = True
            if self.model.diffusion_fitter:
                for p in self.model.diffusion_fitter.parameters(): p.requires_grad = True

    def predict(self, inputs):
        
        '''
        Runs the model on a given set of inputs.
        '''
        
        # run model in eval mode (for batchnorm, dropout, etc.)
        self.model.eval()
        
        return self.model(inputs)
    
    def save(self, save_name):
        
        '''
        Saves model weights and optionally optimizer weights.
        '''
        
        # save model weights
        torch.save(self.model.state_dict(), save_name+'_model')
        
        # save optimizer weights
        if self.save_opt and self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), save_name+'_opt')
        
        # save regularizer weights
        if self.save_reg and self.regularizer is not None:
            torch.save(self.regularizer.state_dict(), save_name+'_reg')
    
    def load(self, 
             model_weights, 
             opt_weights=None, 
             reg_weights=None, 
             device=None):
        
        '''
        Loads model weights and optionally optimizer weights.
        '''
        
        # load model weights
        weights = torch.load(model_weights, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        # load optimizer weights
        if opt_weights is not None:
            params = torch.load(opt_weights, map_location=device)
            self.optimizer.load_state_dict(params)
        
        # load regularizer weights
        if reg_weights is not None:
            params = torch.load(reg_weights, map_location=device)
            self.regularizer.load_state_dict(params)
    
    def load_best_train(self, device=None):
        
        '''
        Loads model weights that yielded best training error.
        '''
        
        # load model weights
        name = self.save_name+'_best_train_model'
        weights = torch.load(name, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        # load optimizer weights
        if self.save_opt and self.optimizer is not None:
            name = self.save_name+'_best_train_opt'
            params = torch.load(name, map_location=device)
            self.optimizer.load_state_dict(params)
        
        # load optimizer weights
        if self.save_reg and self.regularizer is not None:
            name = self.save_name+'_best_train_reg'
            params = torch.load(name, map_location=device)
            self.regularizer.load_state_dict(params)
    
    def load_best_val(self, device=None):
        
        '''
        Loads model weights that yielded best validation error.
        '''
        
        # load model weights
        name = self.save_name+'_best_val_model'
        weights = torch.load(name, map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        # load optimizer weights
        if self.save_opt and self.optimizer is not None:
            name = self.save_name+'_best_val_opt'
            params = torch.load(name, map_location=device)
            self.optimizer.load_state_dict(params)
        
        # load optimizer weights
        if self.save_reg and self.regularizer is not None:
            name = self.save_name+'_best_val_reg'
            params = torch.load(name, map_location=device)
            self.regularizer.load_state_dict(params)