import torch
import torch.nn as nn

def normalize_sets(x_train, y_train, x_val, y_val):
    train = torch.cat([x_train, y_train], dim=1)
    train_shapes = [x_train.shape[1], y_train.shape[1]]
    
    val =  torch.cat([x_val, y_val], dim=1)
    val_shapes = [x_val.shape[1], y_val.shape[1]]
    
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)

    train_norm = (train - mu) / sigma
    val_norm = (val - mu) / sigma
    
    x_train_norm, y_train_norm = torch.split(train_norm, train_shapes, dim=1)
    x_val_norm, y_val_norm = torch.split(val_norm, val_shapes, dim=1)
    
    return x_train_norm, y_train_norm, x_val_norm, y_val_norm, mu, sigma

def normalize(denorm_data, mu, sigma):
    norm_data = (denorm_data - mu) / sigma 
    return norm_data

def denormalize(norm_data, mu, sigma):
    # ReLU to prevent negative concentrations
    denorm_data = torch.relu(norm_data * sigma + mu)
    return denorm_data