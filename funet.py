from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
import torch

def build_res_unet(n_input=1, n_output=2, size=256):
    '''
    Creates a generator by building a Unet model with a Resnet18 backbone
    
    Args:
      n_input (int): Corresponds to the L feature dimension in L*a*b* color space
      n_output (int): Corresponds to the ab feature dimensions in L*a*b* color space
      size (int): height/width of image
      
    Returns:
      Generator model
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G