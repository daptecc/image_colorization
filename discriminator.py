from torch import nn

class PatchDiscriminator(nn.Module):
    '''
    Discriminator that predicts whether an image in L*a*b* color space is real or fake
    '''
    
    def __init__(self, input_c, num_filters=64, n_down=3):
        '''
        Creates a discriminator model consisting of conv/batchnorm/leakyrelu blocks
        Default parameters creates three blocks with filters with the progression (64, 128), (128, 256), (256, 512), and a final block (512, 1)
        
        Args:
          input_c (int): input number of channels
          num_filters (int): number of filters for starting block
          n_down (int): number of blocks to construct
        '''
        
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        
        # the 'if' statement is taking care of not using stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2) for i in range(n_down)]
        
        # make sure to not use normalization or activation for the last layer of the model
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        
        self.model = nn.Sequential(*model)                                                   
        
        
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        '''
        Construct a block of layers when needing to make some repetitive blocks of layers consisting of conv2d, and optional batchnorm2d, optional leakyReLU
        
        Args:
          ni (int): input channel for conv2d
          nf (int): output channel for conv2d
          k (int): kernel size for conv2d
          s (int): stride for conv2d
          p (int): padding for conv2d
          norm (bool): include batchnorm2d
          act (bool): include leakyReLU activation
         
        Returns:
          nn.Sequential layers
        '''
        
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
