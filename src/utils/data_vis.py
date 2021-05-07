from shapely.geometry import box
from utils import data_prep as dp
from utils import data_vis as dv
from utils import helpers as h

import rasterio
import rasterio.mask
import rasterio.plot as rp
import time
import glob
import os
import numpy as np
# import cartopy.crs as ccrs
# import metpy
if ("SSH_CONNECTION" in os.environ) or ('SSH_TTY' in os.environ):
    import matplotlib
    # dont display plot if on remote server
    matplotlib.use('agg')

import matplotlib.pyplot as plt
# plt.switch_backend('agg')

def show_map_image(proj, smoke_plume_df=None, save_file_path=None, axis=None, bounds=None, smoke_bbox=True):
    """[summary]

    Arguments:
        proj {[type]} -- [description]

    Keyword Arguments:
        smoke_plume_df {[type]} -- [description] (default: {None})
        save_file_path {[type]} -- [description] (default: {None})
        axis {[type]} -- [description] (default: {None})
        bounds {iterable list or tuple} -- bounds in latlong (xmin, ymin, xmax, ymax)
        smoke_bbox {bool} -- return bbox instead of polygons on image (default: {True})
    """
    if axis is not None:
        pass

    else:

        DPI = 300

        # create a two panel figure: one with no enhancement, one using sqrt()
        fig = plt.figure(figsize=(15, 15), frameon=False)
        #fig = plt.figure()

        # create axis with Geostationary projection
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # set image bounds
        if bounds is not None:
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        if smoke_plume_df is not None:

            for row in smoke_plume_df.itertuples():
                
                if smoke_bbox:
                    ## get the bounds of the smoke geometry
                    smoke_bounds = row.geometry.bounds 
                    ax.add_geometries([box(smoke_bounds[0],smoke_bounds[1],smoke_bounds[2],smoke_bounds[3])], 
                                    crs=ccrs.PlateCarree(), facecolor='none', edgecolor='blue')
                
                else:
                    ax.add_geometries([row.geometry], ccrs.PlateCarree(), alpha=0.5)

        #ax.coastlines(resolution='50m', color='black', linewidth=0.2)
        ax.add_feature(ccrs.cartopy.feature.STATES, linewidth=0.2)

        ax.set_axis_off()

        try:
            # plot image and calculate max range for colormap
            vmax = np.sqrt(proj.max_rad)
            np.sqrt(proj).plot.imshow(ax=ax, x='x', y='y', vmin=0, vmax=vmax, cmap='Greys_r', transform=proj.metpy.cartopy_crs,
                                      add_colorbar=False, add_labels=False)
        except AttributeError:
            np.sqrt(proj).plot.imshow(ax=ax, x='x', y='y', cmap='Greys_r', transform=proj.metpy.cartopy_crs,
                                      add_colorbar=False, add_labels=False)

        if save_file_path is not None:
            fig.savefig(save_file_path, dpi=DPI, bbox_inches=None, transparent=True, edgecolor=None, )
        # vmax = np.power(proj.max_rad, 1/gamma_correction)
        # np.power(proj,1/gamma_correction).plot.imshow(ax=ax, x='x', y='y', vmin=0, vmax=vmax, cmap='Greys_r', transform=proj.metpy.cartopy_crs, add_colorbar=False)
        # vmax = proj.max_rad
        # proj.plot.imshow(ax=ax, x='x', y='y', vmin=0, vmax=vmax, cmap='Greys_r', transform=proj.metpy.cartopy_crs, add_colorbar=False)

## adapted from https://stackoverflow.com/a/51438809
def show_figure(arr, save_file_path=None, dpi=192, resize_fact=1, ax=None):
    """
    Export array as figure in original resolution
    Args:
        arr: array of image to save in original resolution
        save_file_path: name of file where to save figure
        dpi: resize facter wrt shape of arr, in (0, np.infty)
        resize_fact: dpi of your screen
        ax: axes for graph

    Returns: plotted figure

    """

    if len(arr.shape) == 2:
        reshaped_arr = arr
    else:
        reshaped_arr = rp.reshape_as_image(arr)

        if (reshaped_arr.shape[2] == 2):
            reshaped_arr = reshaped_arr[:,:,0]

    if ax is not None:
        ax.axis('off')
        ax.imshow(reshaped_arr, cmap='Greys_r')

    else:
        plt.close()

        fig = plt.figure(frameon=False)
        fig.set_size_inches(reshaped_arr.shape[1] / dpi, reshaped_arr.shape[0] / dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(reshaped_arr, cmap='Greys_r')

    if save_file_path is not None:
        plt.savefig(save_file_path, dpi=(dpi * resize_fact))

def generate_smoke_mask(smoke_plume_df, src_file_name, save_data_path=None, ax=None):
    """
    Create binary smoke mask that is paired to the .tiff images
    Args:
        smoke_plume_df: smoke plume dataframe
        src_file_name: filename for source geo .tiff image with transformation information
        save_data_path: path to save the mask at

    Returns: save image if filename is given

    """
    # open the geo .tiff file
    src = rasterio.open(src_file_name)

    # get filter parameters from filename string
    param_dict = h.get_true_color_filename_split_dict(src_file_name)

    # filter the smoke plume data to what we need
    filtered_smoke_df = h.filter_plumes(smoke_plume_df, param_dict)

    smoke_df_geoms = filtered_smoke_df['geometry'].values

    # create rasters from shapes files
    mask = rasterio.features.rasterize(shapes=smoke_df_geoms, out_shape=(src.meta['height'], src.meta['width']), fill=0,
                                       transform=src.meta['transform'])

    if save_data_path is not None:

        if ax is not None:
            ax.axis('off')
            show_figure(mask, save_data_path, ax=ax)
        else:
            show_figure(mask, save_data_path)

    else:
        if ax is not None:
            ax.axis('off')
            show_figure(mask, ax=ax)
        else:
            show_figure(mask)

def generate_smoke_mask_wrapper(file_name, smoke_plume_df):
    """
    Create smoke mask for a given file and return the mask file name. Used in an apply function
    Args:
        file_name: true color file name
        smoke_plume_df: plume data

    Returns:
        string of the mask filename
    """

    # create filenames to be stored in the index file by parsing true_color filename
    temp_file_name = '_'.join(file_name.split('_')[2:])

    # catch error with there are no polygons to plot (due to some of the previous filtering logic...)
    try:
        # generate smoke mask
        dv.generate_smoke_mask(smoke_plume_df,
                               '../data/img/{}'.format(file_name),
                               save_data_path='../data/mask/{}'.format('mask_' + temp_file_name))

    except ValueError as e:
        print(e)

    return ('mask_' + temp_file_name)

def show_img_mask(src_file_name, smoke_plume_df, bands=['true_color'], save_data_path=None, labels=False):
    """
    Create a vector of images showing true color and additional bands provided as well as the mask
    Args:
        src_file_name: true color file name
        smoke_plume_df: plume data
        bands: list of bands that should be plotted (at least must include true color)
        save_data_path: file to save data to
        labels: whether or not to add labels to the graph

    Returns:

    """

    f, ax = plt.subplots(1, len(bands) + 1, figsize=(14, 20 / (len(bands) + 1)))
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    ax = ax.ravel()

    for idx in range(len(bands)):
        file_name = bands[idx] + '_' + '_'.join(src_file_name.split('_')[2:])

        if file_name[-4:] != 'tiff':
            file_name = glob.glob('../data/img/{}*.tiff'.format(file_name))[0].split('/')[-1]

        # open the geo .tiff file
        src = rasterio.open('../data/img/{}'.format(file_name))

        if labels:
            # show labels
            ax[idx].set_title(bands[idx])

        # plot image
        dv.show_figure(src.read(), ax=ax[idx])

    # plot the mask at the end as the last image
    if labels:
        ax[-1].set_title('mask')

    if file_name.split('_')[0] != 'true':
        tc_file_name = '../data/img/true_color_{}'.format('_'.join(file_name.split('_')[1:]))
    else:
        tc_file_name = '../data/img/{}'.format(file_name)

    # dv.generate_smoke_mask(smoke_plume_df, '../data/img/{}'.format(src_file_name), ax=ax[-1])
    dv.generate_smoke_mask(smoke_plume_df, tc_file_name, ax=ax[-1])

    if save_data_path is not None:
        f.savefig(save_data_path, transparent=True)

