from torch import nn

def init_weights(net, init='norm', gain=0.02):
    '''
    Initializes conv layer and batchnorm weights of the model
    
    Args:
      init (str): option to initial weights using normal, xavier, or kaiming initialization for conv layers
      gain (float): value for normal initialization of batchnorm weights
      
    Returns:
      model with initialized weights
    '''
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    '''
    Initalizes the weights of the model
    '''
    model = model.to(device)
    model = init_weights(model)
    return model
