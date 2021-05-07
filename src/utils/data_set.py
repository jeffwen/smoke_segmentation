## NOT CURRENTLY IN USE ##

import numpy as np
import glob
from .helpers import glob_files
from skimage import io
from torch.utils.data.dataset import Dataset

class WildfireSmokeDataset(Dataset):
    """
    Wildfire Smoke Dataset class for PyTorch to read in wildfire smoke segmentation data
    """
    def __init__(self, source_dir, train_val_test='train', bands=['true_color','C07','C11'], transform=None):
        """
        Args:
            source_dir: path to the source and target images
            train_val_test: data type to return
            bands: list of bands to use
            transform: transform applied to source and target image
        """
        self.source_dir = source_dir
        self.source_fls = glob_files(f"{self.source_dir}/{train_val_test}/img/{bands[0]}*.png")
        self.target_fls = glob_files(f"{self.source_dir}/{train_val_test}/mask/*.png")
        self.transform = transform

        if (len(self.source_fls) != len(self.target_fls)) or (len(self.source_fls) == 0):
            raise ValueError(f'Number of source and target images must match and there must be a non-zero number of images.')

    def __len__(self):
        return len(self.source_fls)

    def __getitem__(self, idx):
        pass