import torch
from torch import nn, optim
from init import init_model
from gan_loss import GANLoss
from discriminator import PatchDiscriminator
from unet import Unet

class MainModel(nn.Module):
    '''
    Main model consisting of generator that predicts ab features from L input of L*a*b* image and a discriminator that predicts whether the reconstructed L*a*b* image is real or fake
    '''
    
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        '''
        Instantiates generator, generator loss, L1 loss, discriminator, discriminator loss, and optimizers for both the generator and discriminator
        
        Args:
          net_G (nn.Module): optional generator if using pretrained architectures
          lr_G (float): generator learning rate
          lr_D (float): discriminator learning rate
          beta1 (float): Adam optimizer coefficient used for computing running averages of gradient and its square
          beta2 (float): Adam optimizer coefficient used for computing running averages of gradient and its square
          lambda_L1 (float): regularization parameter for L1 loss added to the generator loss
        '''
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        '''
        Sets the parameters of the model to require gradient calculation
        '''
        
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        '''
        Puts the data on GPU if available
        '''
        
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        '''
        Generator predicts ab features from L features of L*a*b* images
        '''
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        '''
        Backward pass for the disciminator 
        
        Predicted L*a*b* image is reconstructed from predicted ab features and real L input features
        Predicted image is fed into discriminator to get fake prediction
        Discriminator loss for fake image is computed by comparing fake prediction to fake labels
        Real L*a*b* image is reconstructed from real ab features and real L input features
        Real image is fed into discriminator to get real prediction
        Discriminator loss for real image is computed by comparing real prediction to real labels
        Final discriminator loss is computed by taking the average of real and fake discriminator losses
        '''
        
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        '''
        Backward pass for the generator
        
        Predicted L*a*b* image is reconstructed from predicted ab features and real L input features
        Predicted image is fed into discriminator to get fake prediction
        Generator loss for fake image is computed by comparing fake prediction to real labels
        L1 loss computed for predicted ab features and real ab features, with regularization
        Final generator loss is the sum of generator loss and L1 loss
        '''
        
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        '''
        Perform one step of forward/backward passes for generator and discriminator
        '''
        
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

model = MainModel()
