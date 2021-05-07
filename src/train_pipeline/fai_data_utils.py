from fastai import vision as faiv
from fastai import torch_core as tc
from typing import Tuple
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../utils')

from utils import errors as e
import PIL as pil
import math
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_mask(file_name):
    return str(file_name).replace('true_color', 'mask')

def keep_true_color(file_name):
    return ('true_color' in file_name.name)

def show_interp_xyz(learner, interp_obj=None, idx=0, data_type='valid'):

    if (interp_obj is not None) and (data_type=='valid'):
        fig, axs = plt.subplots(1, 3, figsize=(12,10))
        faiv.Image(interp_obj.preds[idx]).show(ax=axs[2], cmap='viridis')
        data_bunch = learner.data.valid_ds[idx]
        data_bunch[0].show(ax=axs[0])
        data_bunch[1].show(ax=axs[1], cmap='viridis', alpha=1)

        plt.tight_layout()

    elif (interp_obj is None) and (data_type=='train'):
        fig, axs = plt.subplots(1, 2, figsize=(12,10))
        data_bunch = learner.data.train_ds[idx]
        data_bunch[0].show(ax=axs[0])
        data_bunch[1].show(ax=axs[1], cmap='viridis', alpha=1)

        plt.tight_layout()

    else:
        raise e.UnsupportedType("Plotting prediction from interpretation object only available for 'valid' data_type")

class SegmentationChannelLabelList(faiv.SegmentationLabelList):
    def open(self, file_name):
        # mask = faiv.image.open_mask(file_name, div=True, convert_mode='L')
        # px = mask.px
        # new_px = torch.zeros((2, *px.shape[-2:])).float()
        # new_px[0][px[0] == 0] = 1.0
        # new_px[1][px[0] == 1] = 1.0
        # mask.px = new_px
        return faiv.image.open_mask(file_name, div=True)

    def analyze_pred(self, pred, thresh=0.3, **kwargs):

        if pred.shape[0] > 1:
            return pred.argmax(dim=0)[None]
        else:
            px = pred
            new_px = torch.zeros((1, *px.shape[-2:])).float()
            new_px[px > thresh] = 1.0

            return new_px

class SegmentationChannelList(faiv.SegmentationItemList):
    _label_cls, _square_show_res = SegmentationChannelLabelList, False

    def __init__(self, items, **kwargs):
        super().__init__(items, **kwargs)

    @classmethod
    def from_folders(cls, path, bands=['true_color', 'C07', 'C11'], **kwargs):
        res = super().from_folder(path, **kwargs)
        cls.bands = bands
        return res

    def open(self, file_name):

        tensor_list = []

        # storing this so that it doesnt cause issues when reloading this data..
        self.bands = self.bands

        for band in self.bands:

            if band == 'true_color':
                temp_band = faiv.open_image(file_name, convert_mode='RGB', div=True)
            else:
                temp_band = faiv.open_image(file_name, convert_mode='L', div=True)

            tensor_list.append(temp_band.data.float())

        merged_tensor = torch.cat(tensor_list, dim=0)

        return faiv.Image(merged_tensor)

    def show_xys(self, xs, ys, figsize=(8, 8), **kwargs):

        rows = int(math.sqrt(len(xs)))

        fig, axs = plt.subplots(rows, rows, figsize=figsize)

        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            ax.axis('off')

            # for opening targets as 2 channel
            # y = faiv.Image(ys[i].data[1:2, :, :]), alpha = 0

            faiv.Image(xs[i].data[0:3, :, :]).show(ax=ax, y=ys[i], alpha=0.3, **kwargs)

        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize=(8, 8), **kwargs):

        figsize = faiv.ifnone(figsize, (12,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)

        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            faiv.Image(x.data[0:3,:,:]).show(ax=axs[i, 0], y=y, **kwargs)
            # faiv.Image(x.data[0:3, :, :]).show(ax=axs[i, 1], y=z, **kwargs)
            z.show(ax=axs[i, 1], cmap='viridis', alpha=1, **kwargs)