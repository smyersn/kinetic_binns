import torch, time, sys, pdb
import numpy as np

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
            train_loader,
            val_loader,
            device='cpu',
            batch_size=None,
            epochs=1,
            initial_epoch=0,
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
        
        # Create trivial mask for pruning
        mask_shape = self.model.reaction.eql_layer.fc.weight.shape
        mask = torch.ones(mask_shape, dtype=torch.float32, device=device)
                    
        # loop over epochs
        for epoch in range(initial_epoch, initial_epoch + epochs):
            epoch_init = time.time()
            #           
            # training step            
            #
            self.train = True
            self.val = False
                    
            self.model.train()
            epoch_start_time = time.time()
            
            # Prune model and print equation
            if epoch > 0 and epoch % 10000 == 0:
                fn = f'{self.dir_name}/equation.txt'
                file = open(fn, 'a')
                
                self.model.prune()
                
                file.write(f'Pruned Equation Epoch {epoch}\n')
                for term in self.model.generate_equation():
                    file.write(f'{term}\n')
                file.write(f'\n')

                file.close()
                
                # Clear momentum and ADAM buffers
                fc_weight = self.model.reaction.eql_layer.fc.weight  # shape: [out_dim, in_dim], or [num_terms] for 1×N
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p is fc_weight:
                            state = self.optimizer.state[p]

                            # Build a mask: 1 where weight ≠ 0, 0 where weight == 0
                            # Will also be used to zero grads for pruned parameters
                            mask = (fc_weight.data != 0).float()

                            exp = state['exp_avg']
                            with torch.no_grad():
                                exp.mul_(mask)
                            exp_sq = state['exp_avg_sq']
                            with torch.no_grad():
                                exp_sq.mul_(mask)
            
            # Create lists for epoch training losses
            train_losses = []
            train_gls_losses = []
            train_pde_losses = []
            train_reg_losses = []
            print(f'epoch init: {time.time() - epoch_init}')
            end_of_loop = None
            # loop over training batches
            for batch_x_train , batch_y_train in train_loader:                     
                batch_init = time.time() 
                if end_of_loop:
                    print(f'loop lag: {batch_init - end_of_loop}')

                # Move to GPU
                batch_x_train = batch_x_train.to(device)
                batch_y_train = batch_y_train.to(device)
                print(f'train batch init: {time.time() - batch_init}')
                
                # computes loss
                def closure():  
                    compute_loss = time.time()                                        
                    # zero out gradients
                    self.optimizer.zero_grad()
                                        
                    # require gradients
                    batch_x_train.requires_grad = True
                    
                    # run the model
                    y_pred = self.model(batch_x_train)
                                        
                    # compute loss and optional regularization
                    train_loss, train_gls_loss, train_pde_loss, train_reg_loss = self.loss(y_pred, batch_y_train)
                               
                    print(f'compute train loss: {time.time() - compute_loss}')  
                    backward_pass = time.time()
                                               
                    # compute backward pass
                    train_loss.backward(retain_graph=True)
                    
                    self.model.reaction.eql_layer.fc.weight.grad.data.mul_(mask)
                                        
                    train_losses.append(train_loss.cpu().detach().numpy() * len(batch_x_train))
                    train_gls_losses.append(train_gls_loss.cpu().detach().numpy() * len(batch_x_train))
                    train_pde_losses.append(train_pde_loss.cpu().detach().numpy() * len(batch_x_train))
                    train_reg_losses.append(train_reg_loss.cpu().detach().numpy() * len(batch_x_train))
                    print(f'train backward pass: {time.time() - backward_pass}') 
                    
                # update model parameters
                if self.scheduler is None:
                    self.optimizer.step(closure=closure)
                else:
                    self.scheduler.step(closure())
                
                cuda_synch = time.time()                               
                # wait for GPU computations to finish
                if batch_x_train.device != torch.device('cpu'):
                    torch.cuda.synchronize()
                print(f'cuda train sync: {time.time() - cuda_synch}')
                end_of_loop = time.time()
                
            book_keeping = time.time()                                                                                          
            # update book keeping for this epoch
            self.train_loss_dict['loss'].append(np.sum(train_losses) / len(train_loader.dataset))
            self.train_loss_dict['gls'].append(np.sum(train_gls_losses) / len(train_loader.dataset))
            self.train_loss_dict['pde'].append(np.sum(train_pde_losses) / len(train_loader.dataset))
            self.train_loss_dict['reg'].append(np.sum(train_reg_losses) / len(train_loader.dataset))
            
            # if train error improved
            rel_diff = (best_train_loss - self.train_loss_dict['loss'][-1])
            rel_diff /= best_train_loss
            if rel_diff > rel_save_thresh:
                
                # update best training loss
                best_train_loss = self.train_loss_dict['loss'][-1]
                
                # optionally save model and optimizer
                if self.save_best_train:
                    self.save(self.save_name+'_best_train')
            print(f'train book keeping: {time.time() - book_keeping}')
                
            #
            # validation step
            #                
            self.train = False
            self.val = True
            
            self.model.eval()
            
            # Create lists for epoch training losses
            val_losses = []
            val_gls_losses = []
            val_pde_losses = []
            val_reg_losses = []
            
            # loop over validation batches
            for batch_x_val, batch_y_val in val_loader:     
                # Move to GPU
                batch_x_val = batch_x_val.to(device)
                batch_y_val = batch_y_val.to(device)
           
                self.optimizer.zero_grad()
                                                
                # require gradients
                batch_x_val.requires_grad = True
                                
                # run the model
                y_pred = self.model(batch_x_val).data
                
                # comptue loss
                val_loss, val_gls_loss, val_pde_loss, val_reg_loss = self.loss(y_pred, batch_y_val)
                
                val_losses.append(val_loss.cpu().detach().numpy() * len(batch_x_train))
                val_gls_losses.append(val_gls_loss.cpu().detach().numpy() * len(batch_x_train))
                val_pde_losses.append(val_pde_loss.cpu().detach().numpy() * len(batch_x_train))
                val_reg_losses.append(val_reg_loss.cpu().detach().numpy() * len(batch_x_train))
                
                # wait for GPU computations to finish
                if batch_x_val.device != torch.device('cpu'):
                    torch.cuda.synchronize()
                                                   
            # update book keeping for this epoch
            self.val_loss_dict['loss'].append(np.sum(val_losses) / len(val_loader.dataset))
            self.val_loss_dict['gls'].append(np.sum(val_gls_losses) / len(val_loader.dataset))
            self.val_loss_dict['pde'].append(np.sum(val_pde_losses) / len(val_loader.dataset))
            self.val_loss_dict['reg'].append(np.sum(val_reg_losses) / len(val_loader.dataset))

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
                total_iter=initial_epoch+epochs,
                start_time=start_time,
                previous_time=epoch_start_time,
                ops_per_iter=batch_size)
            
            print(f'total epoch length: {time.time() - epoch_init}')

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
