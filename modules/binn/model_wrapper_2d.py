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
        augmentation (callable): Augmentation that inputs x and y.
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
        verbose              (int): How much info to print to the screen:
                                        0: No updates
                                        1: Update after every epoch
                                        2: Update after every batch
        validation_data     (list): Input and target validation data.
        shuffle             (bool): Whether to shuffle data each epoch.
        class_weight        (list): NOT IMPLEMENTED
        sample_weight       (list): NOT IMPLEMENTED
        initial_epoch        (int): Initial epoch to start training.
        steps_per_epoch      (int): Train batches before moving to next epoch.
        validation_steps     (int): Val batches before moving to next epoch.
        validation_freq      (int): NOT IMPLEMENTED
        early_stopping       (int): Number of epochs since validation improved.
        best_train_loss    (float): Best loss on training set.
        best_val_loss      (float): Best loss on validation set.
        include_val_aug     (bool): Inclusion of augmentation in val step.
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
                 regularizer=None,
                 augmentation=None,
                 scheduler=None,
                 save_name=None,
                 save_best_train=False,
                 save_best_val=True,
                 save_opt=False,
                 save_reg=False):
        
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer
        self.augmentation = augmentation
        self.scheduler = scheduler
        self.save_name = save_name
        self.save_best_train = save_best_train
        self.save_best_val = save_best_val
        self.save_opt = save_opt
        self.save_reg = save_reg
        self.train_loss_dict = {'loss': [], 'gls': [], 'pde': []}
        self.val_loss_dict = {'loss': [], 'gls': [], 'pde': []}
        self.train = False
        self.val = False
        
        # if no name specified, don't save weights
        if self.save_name is None:
            self.save_best_train = False
            self.save_best_val = False
            self.save_opt = False
            self.save_reg = False
        
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            early_stopping=None,
            best_train_loss=None,
            best_val_loss=None,
            include_val_aug=False,
            include_val_reg=False,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_save_thresh=0.0,
            ewc=None,
            density_weight=0,
            hist=None,
            edges=None):
        
        # compute train batch size
        if batch_size is None:
            batch_size = len(x)
        train_batches_per_epoch = int(len(x)/batch_size)
        if train_batches_per_epoch == 0:
            train_batches_per_epoch = 1
            batch_size = len(x)        
        
        # compute validation batch size
        if validation_data is not None:
            x_val, y_val = validation_data[0], validation_data[1]
            val_batch_size = batch_size
            val_batches_per_epoch = int(len(x_val)/val_batch_size)
            if val_batches_per_epoch == 0:
                val_batches_per_epoch = 1
                val_batch_size = len(x_val)
        
        # initialize book keeping
        train_length = len(x)
        if validation_data is not None:
            val_length = len(validation_data[0]) 
        start_time = time.time()
        last_improved = 0
        best_train_loss = 1e12 if best_train_loss is None else best_train_loss
        best_val_loss = 1e12 if best_val_loss is None else best_val_loss
        
        # callback at beginning of training
        if callbacks is not None:
            for c in callbacks:
                if c.on_train_begin:
                    c(self)
               
        # loop over epochs
        for epoch in range(initial_epoch, initial_epoch + epochs):
            # print('training')
            #
            # training step
            #
            
            self.train = True
            self.val = False
            
            # callback at beginning of epoch
            if callbacks is not None:
                for c in callbacks:
                    if c.on_epoch_begin:
                        c(self)
            
            self.model.train()
            epoch_start_time = time.time()
            
            # shuffle training data, note the use of .data to detach the 
            # data to prevent the computational graph from growing
            if shuffle:
                p = np.random.permutation(len(x))
                x, y = x[p].data, y[p].data
            
            # loop over training batches
            for idx in range(train_batches_per_epoch):
                # print(f'train: {idx}')
                
                # callback at beginning of batch
                if callbacks is not None:
                    for c in callbacks:
                        if c.on_batch_begin:
                            c(self)
                
                # stop loop if steps_per_epoch exceeded
                if steps_per_epoch is not None:
                    if idx >= steps_per_epoch:
                        break
                
                batch_start_time = time.time()
                
                # computes loss
                def closure():
                    
                    # zero out gradients
                    self.optimizer.zero_grad()
                    
                    # zero-initialize losses
                    self.train_loss = 0
                    self.train_gls_loss = 0
                    self.train_pde_loss = 0
                    self.train_reg_loss = 0

                    # extract input and output batches, NOTE: we use .data to
                    # detach the current batches from their history in order
                    # to prevent the computational graph from growing in memory
                    start = idx * batch_size
                    stop = (idx+1) * batch_size
                    if idx+1 == train_batches_per_epoch:
                        stop = -1
                    x_true = x[start:stop].data.clone()
                    y_true = y[start:stop].data.clone()
                    
                    # optional augmentations
                    if self.augmentation is not None:
                        x_true, y_true = self.augmentation(x_true, y_true)

                    # require gradients
                    x_true.requires_grad = True

                    # run the model
                    y_pred = self.model(x_true)

                    # compute loss and optional regularization
                    train_loss, train_gls_loss, train_pde_loss = self.loss(y_pred, y_true, density_weight, hist, edges)
                    self.train_loss += train_loss
                    self.train_gls_loss += train_gls_loss
                    self.train_pde_loss += train_pde_loss
                    
                    if ewc is not None:
                        penalty = ewc.penalty(self.model)
                        self.train_loss += penalty
                    if self.regularizer is not None:
                        self.train_reg_loss += self.regularizer(self.model, 
                                                                x_true,
                                                                y_true,
                                                                y_pred)
                    self.train_loss += self.train_reg_loss                    

                    # compute backward pass
                    self.train_loss.backward(retain_graph=True)

                    return self.train_loss, self.train_gls_loss, self.train_pde_loss

                # update model parameters
                if self.scheduler is None:
                    self.optimizer.step(closure=closure)
                else:
                    self.scheduler.step(closure())
                
                # update book keeping for this batch
                self.train_loss = self.train_loss.cpu().detach().numpy()
                self.train_gls_loss = self.train_gls_loss.cpu().detach().numpy()
                self.train_pde_loss = self.train_pde_loss.cpu().detach().numpy()
                                
                # wait for GPU computations to finish
                if x.device != torch.device('cpu'):
                    torch.cuda.synchronize()
                
                # print batch statistics
                if verbose == 2:
                    elapsed, remaining, ms_per_iter = time_remaining(
                        current_iter=idx+1, 
                        total_iter=train_batches_per_epoch, 
                        start_time=epoch_start_time, 
                        previous_time=batch_start_time, 
                        ops_per_iter=batch_size)
                    
                    sys.stdout.write(('\r\x1b[KEpoch {0} {1}/{2}' + 
                                    ' | {3} ms ' + 
                                    ' | Train loss = {4:1.4e}' + 
                                    ' | Remaining = ').format(
                        epoch+1,
                        idx+1,
                        train_batches_per_epoch,
                        ms_per_iter,
                        np.mean(self.train_loss)) + remaining + '          ')
                    sys.stdout.flush()
                        
                    # callback at ending of batch
                    if callbacks is not None:
                        for c in callbacks:
                            if c.on_batch_end:
                                c(self)
                                
            # update book keeping for this epoch
            self.train_loss_dict['loss'].append(np.mean(self.train_loss))
            self.train_loss_dict['gls'].append(np.mean(self.train_gls_loss))
            self.train_loss_dict['pde'].append(np.mean(self.train_pde_loss))
            
            # if train error improved
            rel_diff = (best_train_loss - self.train_loss_dict['loss'][-1])
            rel_diff /= best_train_loss
            if rel_diff > rel_save_thresh:
                
                # update best training loss
                best_train_loss = self.train_loss_dict['loss'][-1]
                
                # optionally save model and optimizer
                if self.save_best_train:
                    self.save(self.save_name+'_best_train')

            # print readout
            if verbose == 2:
                sys.stdout.write(('\r\x1b[KEpoch {0}' + 
                                  ' {1}/{2}' + 
                                  ' | {3} ms ' + 
                                  ' | Train loss = {4:1.4e}' + 
                                  ' | Elapsed = ').format(
                    epoch+1,
                    train_batches_per_epoch,
                    train_batches_per_epoch,
                    ms_per_iter,
                    np.mean(self.train_loss_dict['loss'][-1])) + elapsed + '       ')
                sys.stdout.flush()
            
            #
            # validation step
            #
            if validation_data is not None:
                # print('validation')
                
                self.train = False
                self.val = True
                
                self.model.eval()
                
                # zero-initialize losses
                self.val_loss = 0
                self.val_gls_loss = 0
                self.val_pde_loss = 0
                self.val_reg_loss = 0
                
                # loop over validation batches
                for idx in range(val_batches_per_epoch):
                    # print('start of val loop:', flush=True)
                    # print(torch.cuda.memory_summary(abbreviated=True), flush=True)

                    self.optimizer.zero_grad()
                    # callback at beginning of batch
                    if callbacks is not None:
                        for c in callbacks:
                            if c.on_batch_begin:
                                c(self)
                    
                    # stop loop if validation_steps exceeded
                    if validation_steps is not None:
                        if idx >= validation_steps:
                            break                  
                    
                    # extract input and output batches
                    start = idx * val_batch_size
                    stop = (idx+1) * val_batch_size
                    if idx+1 == val_batches_per_epoch:
                        stop = -1
                    x_true = x_val[start:stop].data.clone()
                    y_true = y_val[start:stop].data.clone()
                    # print('after setting batch:', flush=True)
                    # print(torch.cuda.memory_summary(abbreviated=True), flush=True)
                    
                    # require gradients
                    x_true.requires_grad = True
                    
                    # optional augmentations
                    if self.augmentation is not None and include_val_aug:
                        x_true, y_true = self.augmentation(x_true, y_true)
                    
                    # run the model
                    y_pred = self.model(x_true).data
                    # print('after making precition:', flush=True)
                    # print(torch.cuda.memory_summary(abbreviated=True), flush=True)
                    
                    # comptue loss
                    val_loss, val_gls_loss, val_pde_loss = self.loss(y_pred, y_true, density_weight, hist, edges)
                    self.val_loss += val_loss
                    self.val_gls_loss += val_gls_loss
                    self.val_pde_loss += val_pde_loss
                    
                    if ewc is not None:
                        penalty = ewc.penalty(self.model)
                        self.val_loss += penalty                      

                    # optionally include regularization in val loss
                    if include_val_reg and self.regularizer is not None:
                        self.val_reg_loss += self.regularizer(self.model, 
                                                            x_true,
                                                            y_true,
                                                            y_pred)
                    
                    # wait for GPU computations to finish
                    if x.device != torch.device('cpu'):
                        torch.cuda.synchronize()
                        
                    # callback at ending of batch
                    if callbacks is not None:
                        for c in callbacks:
                            if c.on_batch_end:
                                c(self)

                    # print('after calculating loss:', flush=True)
                    # print(torch.cuda.memory_summary(abbreviated=True), flush=True)

    
                # update book keeping for this epoch
                self.val_loss = self.val_loss.cpu().detach().numpy()
                self.val_gls_loss = self.val_gls_loss.cpu().detach().numpy()
                self.val_pde_loss = self.val_pde_loss.cpu().detach().numpy()
                                
                self.val_loss_dict['loss'].append(np.mean(self.val_loss))
                self.val_loss_dict['gls'].append(np.mean(self.val_gls_loss))
                self.val_loss_dict['pde'].append(np.mean(self.val_pde_loss))

                
                # if validation error improved
                rel_diff = (best_val_loss - self.val_loss_dict['loss'][-1])
                rel_diff /= best_val_loss
                if rel_diff > rel_save_thresh:
                    
                    # update best validation loss
                    best_val_loss = self.val_loss_dict['loss'][-1]
                    
                    # optionally save model and optimizer
                    if self.save_best_val:
                        self.save(self.save_name+'_best_val')
                    
                    # update early stopper
                    last_improved = epoch
                    
                    improved = ' *'
                    
                else:
                    
                    improved = ''
            
            # update user
            if verbose == 1:
                
                # times
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
                    if validation_data is not None:
                        p += ' | Val loss = {0:1.4e}'.format(self.val_loss_dict['loss'][-1])
                    p += ' | Remaining = ' + remaining + '           '
                    #sys.stdout.write(p)
                    print(p, flush=True)
                
            # final print readout for verbose 2
            if verbose == 2:
                sys.stdout.write(('\r\x1b[KEpoch {0} {1}/{2}' + 
                                  ' | Train loss = {3:1.4e}' + 
                                  ' | Val loss = {4:1.4e}').format(
                    epoch+1,
                    train_batches_per_epoch,
                    train_batches_per_epoch,
                    self.self.train_loss_dict['loss'][-1],
                    self.val_loss_dict['loss'][-1]))
                print(improved + '                 ')
                
            # optional early stopping
            if early_stopping is not None:
                if epoch - last_improved >= early_stopping:
                    break
                    
            # optional learning rate annealing
            if lr_dec_epoch is not None:
                if np.mod(epoch, lr_dec_epoch) == 0 and epoch != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= lr_dec_prop
                        
            # callback at ending of epoch
            if callbacks is not None:
                for c in callbacks:
                    if c.on_epoch_end:
                        c(self)
                
        # final print readout for verbose 1
        if verbose == 1:
            
            # times
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
            if validation_data is not None:
                p += ' | Val loss = {0:1.4e}'.format(self.val_loss_dict['loss'][idx])
            p += ' | Elapsed = ' + elapsed + '           '
            #sys.stdout.write(p)
            print(p, flush=True)
            
        # callback at ending of training
        if callbacks is not None:
            for c in callbacks:
                if c.on_train_end:
                    c(self)

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
