## NOT CURRENTLY IN USE ##

import numpy as np
import pandas as pd
import glob
import os
from skimage import io
from torch.utils.data.dataset import Dataset

class WildfireSmokeDataset(Dataset):
    """
    Wildfire Smoke Dataset class for PyTorch to read in wildfire smoke segmentation data
    """
    def __init__(self, csv_file, root_dir='crops',train_val_test='train', bands=['true_color','C07','C11'], transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            root_dir (string): 'crops' for image crops
            train_val_test (string): 'train', 'valid', or 'test'
            bands (list): list of bands to use
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train_val_test = train_val_test
        self.root_dir = root_dir
        self.bands = bands
        self.img_path_df = self._filter_df(csv_file).head(25)
        self.transform = transform

    def __len__(self):
        return len(self.img_path_df)

    def __getitem__(self, idx):
        
        # read in true color image
        sat_img_name = os.path.join('../data', self.root_dir, self.train_val_test, self.img_path_df.loc[idx, 'true_color'])
        sat_image = io.imread(sat_img_name)
        
        # read in images
        temp_img_list = []
        for band in self.bands:
            
            temp_img_name = os.path.join('../data', self.root_dir, self.train_val_test, self.img_path_df.loc[idx, band])
            temp_img = io.imread(temp_img_name)

            # add image to dict
            if band == 'true_color':
                temp_img_list.append(temp_img)
                
            elif band in ['C07', 'C11']:
                
                # repeated values for the other bands...
                temp_img_list.append(temp_img[:,:,0])
                
            
        # create numpy image array with all channels
        sat_image = np.dstack(temp_img_list)    

        # read in mask (only binary mask so one channel)
        map_img_name = os.path.join('../data', self.root_dir, self.train_val_test, self.img_path_df.loc[idx, 'mask'])
        map_image = io.imread(map_img_name)[:,:,0]
        
        sample = {'sat_img': sat_image, 'map_img': map_image}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def _filter_df(self, csv_file):
        df = pd.read_csv(csv_file)

        return df[(df['train_val_test'] == self.train_val_test)].reset_index(drop=True)
