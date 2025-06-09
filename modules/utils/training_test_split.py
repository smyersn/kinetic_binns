import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../'
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def to_torch(ndarray, device):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

def training_test_split(training_data, device):
    N = len(training_data)
    p = np.random.permutation(N)
    training_data_shuffled = training_data[p[:]]
    
    # split into train/val and convert to torch
    M = len(training_data_shuffled)
    split = int(0.8*M)
    train_data = to_torch(training_data_shuffled[:split], device)
    val_data = to_torch(training_data_shuffled[split:], device)

    return train_data, val_data

# def training_test_split(training_data, batch_size, species):
#     N = len(training_data)
#     n_train = int(0.8 * N)
#     n_val = int(N - n_train)
    
#     dataset = TensorDataset(torch.from_numpy(training_data)[:, :-species].float(),
#                             torch.from_numpy(training_data)[:, -species:].float())

#     train_ds, val_ds = random_split(
#     dataset,
#     lengths=[n_train, n_val],
#     generator=torch.Generator().manual_seed(42))  # for reproducibility

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,         # shuffle only within the training subset
#         num_workers=8,
#         pin_memory=True,
#         prefetch_factor=3,
#         persistent_workers=True)

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,        # no need to shuffle validation data
#         num_workers=8,
#         pin_memory=True,
#         prefetch_factor=3,
#         persistent_workers=True)

#     return train_loader, val_loader