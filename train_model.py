import glob, time, argparse
from tqdm.notebook import tqdm
import numpy as np
from fastai.data.external import untar_data, URLs
from main_model import MainModel
from colorization_dataset import make_dataloaders
from funet import build_res_unet
from utils import create_loss_meters, update_losses, visualize, log_results

def train_model(model, train_dl, epochs, display_every=200, visualize_dir):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intervals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=True, visualize_dir) # function displaying the model's outputs


if __name__ == '__main__':
     ap = argparse.ArgumentParser()
#     ap.add_argument('-i', '--path', required=False, help='path to image dir')
     ap.add_argument('-g', '--generator', required=False, help='path to pretrained generator')
#     args = vars(ap.parse_args())

    coco_path = untar_data(URLs.COCO_SAMPLE)
    coco_path = str(coco_path) + "/train_sample"
    
    paths = glob.glob(coco_path + "/*.jpg") # Grabbing all the image file names
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]

    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load(args['generator'], map_location=device))
        
    model = MainModel(net_G=net_G)
    
    samples_dir = f'samples_{time.time()}'
    os.mkdir(samples_dir)
        
    train_model(model, train_dl, 100, visualize_dir=samples_dir)
