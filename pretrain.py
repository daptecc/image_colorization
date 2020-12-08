import glob
import numpy as np
from tqdm.notebook import tqdm
from fastai.data.external import untar_data, URLs
import torch
from torch import nn, optim
from funet import build_res_unet
from colorization_dataset import make_dataloaders
from utils import AverageMeter

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    '''
    Pretrain the generator using the L1 loss between predicted and read ab features of images in L*a*b* color space
    
    Args:
      net_G (nn.Module): generator that predicts ab features from L input of L*a*b* image
      train_dl (Dataloader): train dataloader of sampled COCO images
      opt (nn.optim): default is Adam optimizer
      criterion (nn.L1Loss): default is L1 loss
      epochs (int): number of epochs
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


def get_paths():
    '''
    Download sample of COCO dataset
    Sample 10k images from COCO dataset
    Split train/val 80/20
    Return:
    train_paths (list), val_paths (list): image paths
    '''
    coco_path = untar_data(URLs.COCO_SAMPLE)
    coco_path = str(coco_path) + "/train_sample"
    
    paths = glob.glob(coco_path + "/*.jpg") # Grabbing all the image file names
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
    return paths_subset[train_idxs], paths_subset[val_idxs]

if __name__ == '__main__':
    
    train_paths, val_paths = get_paths()
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()        
    pretrain_generator(net_G, train_dl, opt, criterion, 20)
    torch.save(net_G.state_dict(), "res18-unet.pt")