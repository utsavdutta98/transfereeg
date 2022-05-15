import sklearn
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

le = LabelEncoder()

class EEGBiometricDataset(Dataset):

  def __init__(self,subjects,data,frac = 1):
    
    for subject in [88,92,100]:
      if subject in subjects:
          subjects.remove(subject)

    self.subjects = subjects

    X = []
    y = []

    for subject in subjects:
      
      dat = torch.cat(torch.split(torch.Tensor(data[subject][0]),160,dim = 2)[:3])

      if frac != 1:

        indices = random.sample(list(range(dat.shape[0])),int(frac*dat.shape[0]))
        dat = dat[indices]

      X.append(dat)
      y.append([subject-1]*dat.shape[0])

    X = torch.cat(X,dim = 0)
    y = np.concatenate(y,axis = 0)

    y = le.fit_transform(y)

    self.X = X
    self.y = y

  def __getitem__(self,index):

    X = torch.unsqueeze(self.X[index],dim = 0)
    y = self.y[index]

    return X,y
  
  def __len__(self):

    return self.X.shape[0]

  def num_subjects(self):

    return len(self.subjects)
    
def create_dataset(subjects,data,frac):
    
    dataset = EEGBiometricDataset(subjects,data,frac)
    return dataset
    
def create_train_test(dataset,split=0.8):

  train_size = int(split* len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

  return train_dataset,test_dataset

def create_loaders(train_dataset,test_dataset,batch_size = 32):

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, test_loader