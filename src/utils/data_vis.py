from skimage import io
from .augmentation import UnNormalize

import os
import pandas as pd
import numpy as np
import matplotlib
if ("SSH_CONNECTION" in os.environ) or ('SSH_TTY' in os.environ):
    # dont display plot if on remote server
    matplotlib.use('agg')

import matplotlib.pyplot as plt
plt.switch_backend('agg')


# helper function for viewing images
def show_map(sat_img, map_img=None, axis=None):
    """
    Return an image with the shape mask if there is one supplied
    """

    if axis:
        axis.imshow(sat_img)

        if map_img is not None:
            axis.imshow(map_img, alpha=0.5, cmap='gray')

    else:
        plt.imshow(sat_img)

        if map_img is not None:
            plt.imshow(map_img, alpha=0.5, cmap='gray')


# helper function to show a batch
def show_map_batch(sample_batched, img_to_show=3, save_file_path=None, as_numpy=False):
    """
    Show image with map image overlayed for a batch of samples.
    """

    # just select 6 images to show per batch
    sat_img_batch = sample_batched['sat_img'][:img_to_show, :, :, :]
    map_img_batch = sample_batched['map_img'][:img_to_show, :, :, :]
    batch_size = len(sat_img_batch)

    f, ax = plt.subplots(int(np.ceil(batch_size / 3)), 3, figsize=(15, int(np.ceil(batch_size / 3)) * 5))
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    ax = ax.ravel()

    # unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    for i in range(batch_size):
        ax[i].axis('off')
        show_map(sat_img=sat_img_batch.cpu().numpy()[i, :, :, :].transpose((1, 2, 0)),
                 map_img=map_img_batch.cpu().numpy()[i, 0, :, :], axis=ax[i])

    if save_file_path is not None:
        f.savefig(save_file_path)

    if as_numpy:
        f.canvas.draw()
        width, height = f.get_size_inches() * f.get_dpi()
        mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.cla()
        plt.close(f)

        return mplimage

    
def show_tensorboard_image(sat_img, map_img, out_img, save_file_path=None):
    """
    Show images side by side for verification on tensorboard. Takes in torch tensors.
    """
    # show different image from the batch
    batch_size = sat_img.size(0)
    img_num = np.random.randint(batch_size)
    
    if sat_img.size(1) == 6:
        f, ax = plt.subplots(2, 3, figsize=(12, 8))
        f.tight_layout()
        f.subplots_adjust(hspace=.05, wspace=.05)
        ax = ax.ravel()
        
        # plot the true_color and second channel image
        ax[0].imshow(sat_img[img_num,0:3,:,:].cpu().numpy().transpose((1,2,0)))
        ax[0].axis('off')
        ax[1].imshow(sat_img[img_num,3:4,:,:].squeeze().cpu().numpy())
        ax[1].axis('off')
        ax[2].imshow(sat_img[img_num,4:5,:,:].squeeze().cpu().numpy())
        ax[2].axis('off')
        ax[3].imshow(sat_img[img_num,5:6,:,:].squeeze().cpu().numpy())
        ax[3].axis('off')
        ax[4].imshow(map_img[img_num,0,:,:].cpu().numpy())
        ax[4].axis('off')
        ax[5].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
        ax[5].axis('off')
        
    elif sat_img.size(1) == 5:
        f, ax = plt.subplots(1, 5, figsize=(12, 5))
        f.tight_layout()
        f.subplots_adjust(hspace=.05, wspace=.05)
        ax = ax.ravel()
        
        # plot the true_color and second channel image
        ax[0].imshow(sat_img[img_num,0:3,:,:].cpu().numpy().transpose((1,2,0)))
        ax[0].axis('off')
        ax[1].imshow(sat_img[img_num,3:4,:,:].squeeze().cpu().numpy())
        ax[1].axis('off')
        ax[2].imshow(sat_img[img_num,4:5,:,:].squeeze().cpu().numpy())
        ax[2].axis('off')
        ax[3].imshow(map_img[img_num,0,:,:].cpu().numpy())
        ax[3].axis('off')
        ax[4].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
        ax[4].axis('off')
        
    elif sat_img.size(1) == 4:
        f, ax = plt.subplots(1, 4, figsize=(12, 5))
        f.tight_layout()
        f.subplots_adjust(hspace=.05, wspace=.05)
        ax = ax.ravel()

        # plot the true_color and second channel image
        ax[0].imshow(sat_img[img_num,0:3,:,:].cpu().numpy().transpose((1,2,0)))
        ax[0].axis('off')
        ax[1].imshow(sat_img[img_num,3:4,:,:].squeeze().cpu().numpy())
        ax[1].axis('off')
        ax[2].imshow(map_img[img_num,0,:,:].cpu().numpy())
        ax[2].axis('off')
        ax[3].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
        ax[3].axis('off')
        
    else:
        f, ax = plt.subplots(1, 3, figsize=(12, 5))
        f.tight_layout()
        f.subplots_adjust(hspace=.05, wspace=.05)
        ax = ax.ravel()

        # just plot the true_color image
        ax[0].imshow(sat_img[img_num,0:3,:,:].cpu().numpy().transpose((1,2,0)))
        ax[0].axis('off')
        ax[1].imshow(map_img[img_num,0,:,:].cpu().numpy())
        ax[1].axis('off')
        ax[2].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
        ax[2].axis('off')
    
    return(f)

#     if save_file_path is not None:
#         f.savefig(save_file_path)

#     if as_numpy:
#         f.canvas.draw()
#         width, height = f.get_size_inches() * f.get_dpi()
#         mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
#         plt.cla()
#         plt.close(f)

#         return mplimage


# def show_tensorboard_image_old(sat_img, map_img, out_img, save_file_path=None, as_numpy=False):
#     """
#     Show 3 images side by side for verification on tensorboard. Takes in torch tensors.
#     """
#     # show different image from the batch
#     batch_size = sat_img.size(0)
#     img_num = np.random.randint(batch_size)

#     f, ax = plt.subplots(1, 3, figsize=(12, 5))
#     f.tight_layout()
#     f.subplots_adjust(hspace=.05, wspace=.05)
#     ax = ax.ravel()

#     # just plot the true_color image
#     ax[0].imshow(sat_img[img_num,0:3,:,:].cpu().numpy().transpose((1,2,0)))
#     ax[0].axis('off')
#     ax[1].imshow(map_img[img_num,0,:,:].cpu().numpy())
#     ax[1].axis('off')
#     ax[2].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
#     ax[2].axis('off')
    
#     return(f)