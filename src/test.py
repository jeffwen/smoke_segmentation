import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import data_set
from utils import data_vis
from models import unet

import torch
import argparse
import pickle


def main(data_dir, out_file, batch_size, bands, model_path, num_workers):
      
    # set up GPU usage if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define model class and load model
    model = unet.UNetSmall(num_channels=len(bands)+2)
    model_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_dict['state_dict'])
    
    # store generate polygons
    #smoke_dict_list = []
    out_map_store_dict = {'out_map':[], 'fname':[]}
    
    # switch to evaluation mode and don't track gradients
    model.eval()
    with torch.no_grad():  

        test_dataset = data_set.WildfireSmokePredictDataset(data_dir, 
                                bands=bands, 
                                transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        # iterate over data
        for idx, (data, fname) in enumerate(tqdm(test_loader, desc='inference')):

            # make prediction and run through sigmoid
            out_preds_raw = model(data)
            out_probs = torch.nn.functional.sigmoid(out_preds_raw)

            # threshold and turn into mask
            out_maps = (out_probs > 0.3).int().detach().numpy()
            
            # store the prediction maps
            out_map_store_dict['out_map'].append(out_maps)
            out_map_store_dict['fname'].append(fname)
            
    # store output dict as pickle for parallel raster to poly
    with open(f'{out_file}', 'wb') as f:
        pickle.dump(out_map_store_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wildfire Smoke Prediction')
    parser.add_argument('data_dir', metavar='DIR',
                        help='path to directory of test image files')
    parser.add_argument('--out_file', type=str, metavar='PATH',
                        help='path to output file')
    parser.add_argument('-b', '--batch_size', default=3, type=int,
                        metavar='N', 
                        help='mini-batch size (default: 3)')
    parser.add_argument('--bands', nargs="+", default=["true_color", "C07", "C11"],
                        help='input channels')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to model')
    parser.add_argument('--num_workers', default=1, type=int, metavar='N',
                        help='number of workers for data loader')

    args = parser.parse_args()

    # run testing
    main(args.data_dir, args.out_file, batch_size=args.batch_size, bands=args.bands, 
         model_path=args.model, num_workers=args.num_workers)
