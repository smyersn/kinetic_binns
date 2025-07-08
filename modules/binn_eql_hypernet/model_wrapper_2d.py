import torch, time, sys, pdb, os
import numpy as np
import psutil

from modules.utils.time_remaining import *
from modules.binn_eql_hypernet.plot_params import plot_params

def print_memory_usage(str):
    """ Helper function to print process RAM usage. """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{str} | Process RAM: {mem_info.rss / 1024**2:.2f} MB")


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
            k_init=1,
            lambda_init=0.00001,
            epoch_interval=5000,
            batch_size=None,
            epochs=1,
            early_stopping=None,
            best_train_loss=None,
            best_val_loss=None,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_save_thresh=0.0,
            fine_tune=False):
                
        # initialize book keeping
        start_time = time.time()
        last_improved = 0
        best_train_loss = 1e12 if best_train_loss is None else best_train_loss
        best_val_loss = 1e12 if best_val_loss is None else best_val_loss   
        all_ws = []
        all_sigmas = []
        
        def print_equation(path, epoch):
            ws = []
            ps = []
            mus = []
            sigmas = []
            
            for hypernet in self.model.reaction.eql_layer.hypernets:
                w, p, mu, sigma = hypernet(train_data.device, inference=True)
                ws.append(w.item())
                ps.append(p.item())
                mus.append(mu.item())
                sigmas.append(sigma.item())
                            
            fn = f'{path}/equation.txt'
            file = open(fn, 'a')   
                           
            file.write(f'After {epoch} epochs:\n')
            # file.write(f'Ws: {ws}\n')
            # file.write(f'Ps: {ps}\n')
            # file.write(f'Mus: {mus}\n')

            for term in self.model.generate_equation():
                file.write(f'{term}\n')
                
            file.write(f'\n')
            file.close()
            
            return ws, sigmas
     
        # loop over epochs
        for epoch in range(0, epochs):
            if epoch == 0:
                k, lambda_ = 0, 0
            elif epoch == epoch_interval:
                k, lambda_ = k_init, lambda_init
            elif epoch == epoch_interval*2:
                k *= 10
                lambda_ *= 10
            elif epoch == epoch_interval*3:
                k = 1000 
                lambda_ *= 10
                                
            #           
            # training step            
            #
            self.train = True
            self.val = False
                    
            self.model.train()
            epoch_start_time = time.time()
                        
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
                x_true = train_data[idx, :-self.species].data.clone()
                y_true = train_data[idx, -self.species:].data.clone()
                
                # zero out gradients
                self.optimizer.zero_grad()
                                    
                # require gradients
                x_true.requires_grad = True
                
                # run the model
                y_pred = self.model(x_true)
                # print(f'pred: {torch.isnan(y_pred).any()}')
                                    
                # compute loss and optional regularization
                train_loss, train_gls_loss, train_pde_loss, train_reg_loss = self.loss(y_pred, y_true, k, lambda_)
                                                                            
                # compute backward pass and update weights
                train_loss.backward()
                                                    
                self.optimizer.step()
                
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
                x_true = train_data[idx, :-self.species].data.clone()
                y_true = train_data[idx, -self.species:].data.clone()
                           
                self.optimizer.zero_grad()
                                                
                # require gradients
                x_true.requires_grad = True
                                
                # run the model
                y_pred = self.model(x_true).data
                
                # comptue loss
                val_loss, val_gls_loss, val_pde_loss, val_reg_loss = self.loss(y_pred, y_true, k, lambda_)
                
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
            if rel_diff > rel_save_thresh:
                
                # update best validation loss
                best_val_loss = self.val_loss_dict['loss'][-1]
                
                # optionally save model and optimizer
                if self.save_best_val:
                    if fine_tune:
                        self.save(self.save_name+'_best_val_fine_tuned')
                    else:
                        self.save(self.save_name+'_best_val')
                
                # update early stopper
                last_improved = epoch
                
                improved = ' *'
                
            else:
                
                improved = ''
            
            # update user
            elapsed, remaining, ms = time_remaining(
                current_iter=epoch+1,
                total_iter=epochs,
                start_time=start_time,
                previous_time=epoch_start_time,
                ops_per_iter=batch_size)
            
            if epoch % 1000 == 0 or epoch == epochs-1:
                ws, sigmas = print_equation(self.dir_name, epoch)
                all_ws.append(ws)
                all_sigmas.append(sigmas)

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
                # if epoch - last_improved >= early_stopping and epoch >= 10000:
                if epoch - last_improved >= early_stopping:
                    break
                    
            # optional learning rate annealing
            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= lr_dec_prop

        # final print readout
        elapsed, remaining, ms = time_remaining(
            current_iter=epoch+1,
            total_iter=epochs,
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
        
        plot_params(all_ws, all_sigmas, self.dir_name)
                    
        return self.train_loss_dict, self.val_loss_dict
                
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
