import torch 
import torch.nn as nn

class EEGBiometricNet(nn.Module):

  def __init__(self,n_classes):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1,64))
    self.avgpool1 = nn.AvgPool2d(kernel_size = (1,8))
    self.bn1 = nn.BatchNorm2d(16)

    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (32,2))
    self.avgpool2 = nn.AvgPool2d(kernel_size = (1,8))
    self.bn2 = nn.BatchNorm2d(32)

    self.linear1 = nn.Linear(in_features = 1056, out_features = 64)
    self.linear3 = nn.Linear(in_features = 64, out_features = n_classes)

  def forward(self,x):

    x = self.conv1(x)
    x = self.avgpool1(x)
    x = self.bn1(x)
    x = nn.ELU()(x)

    x = self.conv2(x)
    x = torch.squeeze(x,dim = 2)
    x = self.avgpool2(x)
    x = self.bn2(x)
    x = nn.ELU()(x)

    x = nn.Flatten()(x)

    x = self.linear1(x)
    x = nn.Dropout(0.5)(x)
    x = nn.ELU()(x)

    x_last_layer = x
    
    x = self.linear3(x) 

    return x, x_last_layer

class EEGNet(nn.Module):

  def __init__(self , F1, D1, F2 = None , C=21):
    super().__init__()

    if F2 is None:
      F2 = D1 * F1

    # Conv2d, (1,64) filter going over all time series, with F1 such filters
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = F1, kernel_size = (1,64), padding = 'same',bias = False)
    self.bn1 = nn.BatchNorm2d(F1)

    # depthwiseconv2d, each channel 
    self.depthwiseconv2d = nn.Conv2d(in_channels = F1, out_channels = D1 * F1, kernel_size = (C,1), groups = F1, padding = 'valid', bias = False)
    self.bn2 = nn.BatchNorm2d(D1*F1)
    self.dropout1 = nn.Dropout2d(p = 0.5)

    self.elu1 = nn.ELU()

    self.layers = nn.ModuleList()
    for i in range(5):
        self.layers.append(ExpandChannelsBlock(D1*F1))
        
    self.block = nn.Sequential(*self.layers)

    self.avgpool1 = nn.AvgPool2d(kernel_size = (1,4))
    self.dropout2 = nn.Dropout2d(p = 0.5)

    # depthwiseSeparableconv2d
    self.depthconv2d = nn.Conv2d(in_channels = D1 * F1, out_channels = D1 * F1 , kernel_size = (1,16), groups = D1 * F1, padding = 'same', bias = False)
    self.pointwise = nn.Conv2d(in_channels = D1 * F1, out_channels = F2, kernel_size = (1,1), bias = False)

    self.elu2 = nn.ELU()
    self.avgpool2 = nn.AvgPool2d(kernel_size = (1,8))
    
    self.dropout3 = nn.Dropout2d(p = 0.5)
    self.Flatten = nn.Flatten()

    self.linear = nn.LazyLinear(out_features = 2, bias = True)

  def forward(self,x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.depthwiseconv2d(x)
    x = self.bn2(x)
    x = self.dropout1(x)
    x = self.elu1(x)
    x = self.block(x)
    x = self.avgpool1(x)
    x = self.dropout2(x)
    x = self.depthconv2d(x)
    x = self.pointwise(x)
    x = self.elu2(x)
    x = self.avgpool2(x)
    x = self.dropout3(x)
    x = self.Flatten(x)
    x = self.linear(x)

    return x
    
class EEGNet(nn.Module):

  def __init__(self , F1 , D1, F2 = None , C=21):
    super().__init__()

    if F2 is None:
      F2 = D1 * F1

    # Conv2d, (1,64) filter going over all time series, with F1 such filters
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = F1, kernel_size = (1,64), padding = 'same',bias = False)
    self.bn1 = nn.BatchNorm2d(F1)

    # depthwiseconv2d, each channel 
    self.depthwiseconv2d = nn.Conv2d(in_channels = F1, out_channels = D1 * F1, kernel_size = (C,1), groups = F1, padding = 'valid', bias = False)
    self.bn2 = nn.BatchNorm2d(D1*F1)
    self.dropout1 = nn.Dropout2d(p = 0.5)

    self.elu1 = nn.ELU()

    self.layers = nn.ModuleList()
    for i in range(5):
        self.layers.append(ExpandChannelsBlock(D1*F1))
        
    self.block = nn.Sequential(*self.layers)

    self.avgpool1 = nn.AvgPool2d(kernel_size = (1,4))
    self.dropout2 = nn.Dropout2d(p = 0.5)

    # depthwiseSeparableconv2d
    self.depthconv2d = nn.Conv2d(in_channels = D1 * F1, out_channels = D1 * F1 , kernel_size = (1,16), groups = D1 * F1, padding = 'same', bias = False)
    self.pointwise = nn.Conv2d(in_channels = D1 * F1, out_channels = F2, kernel_size = (1,1), bias = False)

    self.elu2 = nn.ELU()
    self.avgpool2 = nn.AvgPool2d(kernel_size = (1,8))
    
    self.dropout3 = nn.Dropout2d(p = 0.5)
    self.Flatten = nn.Flatten()

    self.linear = nn.LazyLinear(out_features = 2, bias = True)

  def forward(self,x):

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.depthwiseconv2d(x)
    x = self.bn2(x)
    x = self.dropout1(x)
    x = self.elu1(x)
    x = self.block(x)
    x = self.avgpool1(x)
    x = self.dropout2(x)
    x = self.depthconv2d(x)
    x = self.pointwise(x)
    x = self.elu2(x)
    x = self.avgpool2(x)
    x = self.dropout3(x)
    x = self.Flatten(x)
    x = self.linear(x)

    return x
    
class EEGNetBlock(nn.Module):

  def __init__(self,F1,D,kernel_size,kernel_size1, gru_input_size, poolsize1 = (1,4), poolsize2 = (1,8), F2 = None, nchans = 64):
    super().__init__()

    if F2 is None:
      F2 = D * F1

    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = F1, kernel_size = kernel_size, padding = 'same', bias = False)
    self.bn1 = nn.BatchNorm2d(F1)

    self.gru = nn.GRU(input_size = gru_input_size, hidden_size = 512, num_layers = 2, bias = False, batch_first = True, bidirectional = True)

    self.depthwiseconv2d = nn.Conv2d(in_channels = 1, out_channels = D * F1, kernel_size = (nchans,1), padding = 'valid', bias = False)
    self.bn2 = nn.BatchNorm2d(D*F1)

    self.avgpool1 = nn.AvgPool2d(kernel_size = poolsize1)
    self.dropout1 = nn.Dropout2d(p = 0.5)

    self.depthconv2d = nn.Conv2d(in_channels = D * F1, out_channels = D * F1 , kernel_size = kernel_size1, groups = D * F1, padding = 'same', bias = False)
    self.pointwise = nn.Conv2d(in_channels = D * F1, out_channels = F2, kernel_size = (1,1), bias = False)
    self.bn3 = nn.BatchNorm2d(F2)
    
    self.avgpool2 = nn.AvgPool2d(kernel_size = poolsize2)
    self.dropout2 = nn.Dropout2d(p = 0.5)
    self.Flatten = nn.Flatten()

  def forward(self,x):
    
    N = x.shape[0]
    ## Conv - BN - ReLU
    x = self.conv1(x)
    x = self.bn1(x)
    x = nn.LeakyReLU(0.01)(x)
    ##
    
    ## GRU 
    x = torch.squeeze(x).view(N,-1,160)
    x = torch.transpose(x,2,1)

    x,_ = self.gru(x)

    x = torch.transpose(x,2,1)
    x = torch.unsqueeze(x,dim = 1)
    
    # ## 
    x = self.depthwiseconv2d(x)
    x = self.bn2(x)
    x = nn.LeakyReLU(0.01)(x)

    x = self.avgpool1(x)
    x = self.dropout1(x)
    x = self.depthconv2d(x)
    x = self.pointwise(x)
    x = self.bn3(x)
    x = nn.LeakyReLU(0.01)(x)

    x = self.avgpool2(x)
    x = self.dropout2(x)
    x = self.Flatten(x)

    return x

class EEGFusionNet(nn.Module):

  def __init__(self):
    super().__init__()

    self.block1 = EEGNetBlock(8, 2, (1,64), (1,8),  512, F2 = 16)
    self.block2 = EEGNetBlock(16,2, (1,96), (1,16), 1024, F2 = 16)
    self.block3 = EEGNetBlock(32,2, (1,128),(1,32), 2048, F2 = 16)
    self.linear = nn.Linear(in_features = 230640, out_features = 2)

  def forward(self,x):

    x1 = self.block1(x)
    x2 = self.block2(x)
    x3 = self.block3(x)

    x = torch.cat((x1,x2,x3),axis = 1)
    x = self.linear(x)
    x = nn.Softmax(dim = -1)(x)
    return x