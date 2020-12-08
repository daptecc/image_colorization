import torch
from torch import nn

class GANLoss(nn.Module):
    '''
    GAN loss for generator
    '''
    
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        '''
        Args:
          gan_mode (str): choose whether to use binary cross entropy loss or mean squared error loss
          real_label (float): value of real label
          fake_label (float): value of fake label
        '''
        
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
    
    def get_labels(self, preds, target_is_real):
        '''
        Helper function to create a vector real or fake labels the same size as preds
        
        Args:
          preds (tensor): tensor of predicted values
          target_is_real (bool): flag to construct vector of real or fake labels
        
        Returns:
          tensor of real or fake labels
        '''
        
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)
    
    def __call__(self, preds, target_is_real):
        '''
        Computes the loss of predictions and real/fake labels
        '''
        
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
