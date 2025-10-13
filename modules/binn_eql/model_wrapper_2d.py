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
            batch_size=None,
            epochs=1,
            initial_epoch=0,
            early_stopping=None,
            best_train_loss=None,
            best_val_loss=None,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_save_thresh=0.0,
            prune_thresh=3,
            warm_up=0):
                
        # initialize book keeping
        start_time = time.time()
        last_improved = 0
        best_train_loss = 1e12 if best_train_loss is None else best_train_loss
        best_val_loss = 1e12 if best_val_loss is None else best_val_loss        
                            
        # loop over epochs
        for epoch in range(initial_epoch, initial_epoch + epochs):
            #           
            # training step            
            #
            self.train = True
            self.val = False
                    
            self.model.train()
            epoch_start_time = time.time()
            
            # # Prune equation every 100 epochs
            # if epoch % 100 == 0 and epoch > 0:
            #     self.model.prune()
            #     # self.freeze_pruned_params()

            # Print equation every 1000 epochs                                                                        
            if epoch % 1000 == 0:  
                fn = f'{self.dir_name}/equation.txt'
                file = open(fn, 'a')
                
                file.write(f'Equation Epoch {epoch}\n')
                for term in self.model.generate_equation():
                    file.write(f'{term}\n')
                file.write(f'\n')

                file.close()

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
                train_loss, train_gls_loss, train_pde_loss, train_reg_loss = self.loss(y_pred, y_true, epoch)
                                                                            
                # compute backward pass and update weights
                train_loss.backward()
                                                    
                self.optimizer.step()
                
                # Clamp parameters between bounds
                with torch.no_grad():
                    bound = self.model.param_bounds
                    self.model.reaction.eql_layer.fc.weight.data.clamp_(
                        -bound, bound)

                    if not self.model.diff_coeffs:
                        self.diffusion_fitter().clamp_(0, bound)
                        
                # Update losses
                train_losses += train_loss.item() * len(x_true)
                train_gls_losses += train_gls_loss.item() * len(x_true)
                train_pde_losses += train_pde_loss.item() * len(x_true)
                train_reg_losses += train_reg_loss.item() * len(x_true)

            # update book keeping for this epoch
            self.train_loss_dict['loss'].append(np.sum(train_losses) / len(train_data))
            self.train_loss_dict['gls'].append(np.sum(train_gls_losses) / len(train_data))
            self.train_loss_dict['pde'].append(np.sum(train_pde_losses) / len(train_data))
            self.train_loss_dict['reg'].append(np.sum(train_reg_losses) / len(train_data))
            
            # if train error improved
            rel_diff = (best_train_loss - self.train_loss_dict['loss'][-1])
            rel_diff /= best_train_loss
            if rel_diff > rel_save_thresh:
                
                # update best training loss
                best_train_loss = self.train_loss_dict['loss'][-1]
                
                # optionally save model and optimizer
                if self.save_best_train:
                    # print(f'Pruned and saved at epoch {epoch}')
                    # self.model.prune(thresh=prune_thresh)
                    # self.freeze_pruned_params()
                    self.save(self.save_name+'_best_train')

            #
            # validation step
            #                
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
                val_loss, val_gls_loss, val_pde_loss, val_reg_loss = self.loss(y_pred, y_true, epoch)
                
                val_losses += val_loss.item() * len(x_true)
                val_gls_losses += val_gls_loss.item() * len(x_true)
                val_pde_losses += val_pde_loss.item() * len(x_true)
                val_reg_losses += val_reg_loss.item() * len(x_true)

            # update book keeping for this epoch
            self.val_loss_dict['loss'].append(np.sum(val_losses) / len(val_data))
            self.val_loss_dict['gls'].append(np.sum(val_gls_losses) / len(val_data))
            self.val_loss_dict['pde'].append(np.sum(val_pde_losses) / len(val_data))
            self.val_loss_dict['reg'].append(np.sum(val_reg_losses) / len(val_data))

            # if validation error improved
            rel_diff = (best_val_loss - self.val_loss_dict['loss'][-1])
            rel_diff /= best_val_loss
            
            if epoch >= warm_up*2:
                if rel_diff > rel_save_thresh:
                    
                    # update best validation loss
                    best_val_loss = self.val_loss_dict['loss'][-1]
                    
                    # optionally save model and optimizer
                    if self.save_best_val:
                        # print(f'Pruned and saved at epoch {epoch}')
                        # self.model.prune(thresh=prune_thresh)
                        # self.freeze_pruned_params()
                        self.save(self.save_name+'_best_val')
                    
                    # update early stopper
                    last_improved = epoch
            
            else:
                last_improved = epoch
                
                                
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
                
            # optional early stopping
            if early_stopping is not None:
                # if epoch - last_improved >= early_stopping and epoch > 50000:
                if epoch - last_improved >= early_stopping:
                    break
                    
            # optional learning rate annealing
            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= lr_dec_prop

        # # final prune
        # if self.save_best_train:
        #     self.load(self.save_name+'_best_train_model')
        #     self.model.prune(thresh=prune_thresh)
        #     self.save(self.save_name+'_best_train')

        # if self.save_best_val:
        #     self.load(self.save_name+'_best_val_model')
        #     self.model.prune(thresh=prune_thresh)
        #     self.save(self.save_name+'_best_val')
            
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
            
        return self.train_loss_dict, self.val_loss_dict
    
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