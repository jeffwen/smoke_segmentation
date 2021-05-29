from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils import helpers as h

import numpy as np
import pandas as pd
import torch
import glob
import os

# some images dont load properly, but dont seem to have problems
# doing this to get around issue
ImageFile.LOAD_TRUNCATED_IMAGES = True

class WildfireSmokeDataset(Dataset):
    """
    Wildfire Smoke Dataset class for PyTorch to read in wildfire smoke segmentation data
    """
    def __init__(self, csv_file, root_dir='crops', train_val_test='train', 
                 bands=['true_color','C07','C11'], transform=None, multitask=False):
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
        self.img_path_df = self._filter_df(csv_file)
        self.transform = transform
        self.multitask = multitask

    def __len__(self):
        return len(self.img_path_df)

    def __getitem__(self, idx):
        
        # read in true color image
        #sat_img_name = os.path.join('../data', self.root_dir, self.train_val_test, self.img_path_df.loc[idx, 'true_color'])
        #sat_image = io.imread(sat_img_name)
        #sat_image = np.array(Image.open(sat_img_name))        

        # read in images
        temp_img_list = []
        for band in self.bands:
            
            temp_img_name = os.path.join('../data', self.root_dir, self.train_val_test, self.img_path_df.loc[idx, band])
            
            if band == 'merra2':
                try:
                    # clip values to between 0 and 5 for merra2
                    merra2_img = np.clip(np.load(temp_img_name), a_min=0, a_max=5)
                except ValueError:
                    print(f"error reading: {temp_img_name}")
                
            else:
                temp_img = np.array(Image.open(temp_img_name))

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
        map_image = np.array(Image.open(map_img_name))[:,:,0]        

        if self.multitask:
            sample = {'sat_img': sat_image, 'map_img': map_image, 'aod_img': aod_image}
        else:
            sample = {'sat_img': sat_image, 'map_img': map_image}
        
        if self.transform:
            sample = self.transform(sample)
            
            # transfrom merra2 separately cause of different scales
            if 'merra2' in self.bands:
                #merra2_img = transforms.functional.to_tensor(merra2_img)
                merra2_img = transforms.functional.normalize(torch.as_tensor(merra2_img).unsqueeze(0), mean=[0.13749852776527405], std=[0.042889975011348724])
                
                # concatenate merra2 onto sat image
                sample['sat_img'] = torch.cat((sample['sat_img'], merra2_img), dim=0)

        return sample
    
    def _filter_df(self, csv_file):
        df = pd.read_csv(csv_file)

        return df[(df['train_val_test'] == self.train_val_test)].reset_index(drop=True)
    

class WildfireSmokePredictDataset(Dataset):
    """
    Wildfire Smoke Dataset class for PyTorch to read in wildfire smoke segmentation data
    """
    def __init__(self, files_dir, bands=['true_color','C07','C11'], transform=transforms.ToTensor()):
        """
        Args:
            files_dir (string): Path to image files
            bands (list): list of bands to use
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir = files_dir
        self.bands = bands
        self.transform = transform
        self.img_dirname, self.img_slugs = self._find_imgs(files_dir)


    def __len__(self):
        return len(self.img_slugs)

    def __getitem__(self, idx):
        
        # read in images
        temp_img_list = []
        for band in self.bands:
                
            if band == 'merra2':
                try:

                    # get merra filename params from true color slug
                    fname_params = h.get_true_color_filename_split_dict(self.img_slugs[idx])
                    #true_color_fn = file_str.split('/')[-1]
                    #start_end_slug = true_color_fn.split('_')[4] +'_'+ true_color_fn.split('_')[5].split('.')[0]

                    temp_img_name = f"{self.img_dirname}/{band}_{fname_params['year']}{fname_params['month']}{fname_params['day']}.tiff"

                    # clip values to between 0 and 5 for merra2
                    merra2_img = np.clip(np.array(Image.open(temp_img_name)), a_min=0, a_max=5)

                except ValueError:
                    print(f"error reading: {temp_img_name}")
                
            else:
                
                # get image filename from true color slug
                temp_img_name = f"{self.img_dirname}/{band}_{'_'.join(self.img_slugs[idx].split('/')[-1].split('_')[2:])}"
                temp_img = np.array(Image.open(temp_img_name))

                # add image to dict
                if band == 'true_color':
                    temp_img_list.append(temp_img[:,:,0:3])

                elif band in ['C07', 'C11']:

                    # repeated values for the other bands...
                    temp_img_list.append(temp_img[:,:,0])
                
            
        # create numpy image array with all channels
        sat_image = np.dstack(temp_img_list)     
        
        if self.transform:
            sat_image = self.transform(sat_image)
            
            # transfrom merra2 separately cause of different scales
            if 'merra2' in self.bands:
                
                merra2_img = transforms.functional.normalize(torch.as_tensor(merra2_img).unsqueeze(0), mean=[0.13749852776527405], std=[0.042889975011348724])
                
                # concatenate merra2 onto sat image
                sat_image = torch.cat((sat_image, merra2_img), dim=0)

        return sat_image, self.img_slugs[idx]
    
    def _find_imgs(self, files_dir):

        ## gather image filenames for true_color (which each has corresponding channel/ masks)
        imgs = glob.glob(f'{files_dir}/true_color*')
        img_dirname = os.path.dirname(imgs[0])
        img_slugs = sorted([os.path.basename(file_str) for file_str in imgs])

        return img_dirname, img_slugs
