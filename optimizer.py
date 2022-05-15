import torch

def build_optimizer(model,optimizer, lr = None, weight_decay = 1e-5):

  if optimizer == 'adam':

    if lr is not None:
      return torch.optim.Adam(model.parameters(),lr = lr, weight_decay = weight_decay)
    else:
      return torch.optim.Adam(model.parameters(), weight_decay = weight_decay)

  if optimizer == 'sgd':
    
    if lr is not None:
      return torch.optim.SGD(model.parameters(),lr = lr, weight_decay = weight_decay)
    else:
      return torch.optim.SGD(model.parameters(), weight_decay = weight_decay)

  if optimizer == 'rmsprop':

    if lr is not None:
      return torch.optim.RMSprop(model.parameters(),lr = lr, weight_decay = weight_decay)
    else:
      return torch.optim.RMSprop(model.parameters(), weight_decay = weight_decay)

  else:

    raise ValueError('Enter a valid optimizer')