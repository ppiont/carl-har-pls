import numpy as np
import pdb
from torch.utils.data import DataLoader
import torch.optim
import torch

#Define a dataset class to be used for a dataloader
class Dataset(torch.utils.data.Dataset):
  """Characterizes a dataset for PyTorch"""
  def __init__(self, features, targets):
    """Initialization"""
    self.features = features
    self.targets = targets

  def __len__(self):
    """Denotes the total number of samples"""
    return len(self.targets)

  def __getitem__(self, index):
    """Generates one sample of data"""
    return self.features[index], self.targets[index]



def create_data_loader(features, targets,
                       batch_size, shuffle, drop_last):
  '''Create dataloader'''

  # Create torch training dataset
  dataset = Dataset(torch.Tensor(features),
                          torch.Tensor(targets))

  # Create dataloader for training data
  data_loader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)

  return data_loader
