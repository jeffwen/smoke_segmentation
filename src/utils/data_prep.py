from datetime import datetime
from pyresample.geometry import AreaDefinition
from utils import helpers as h
from skimage import io
from PIL import Image
from tqdm import tqdm
from rasterio.features import shapes

import shapely
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import satpy as sp
import random
import json
import os

def year_month_day(datetime_fullstr, datetime_substr):
    """Function to extract the year, month, day, start time, end time, day of year from smoke plume data 

    Arguments:
        datetime_fullstr {str} -- String in date time format %Y%j
        datetime_substr {str} -- Date time string of the component that should be returned (e.g. %Y, %m, %d)
    
    Returns:
        str -- substring based on the datetime_substr provided above
    """
    if (datetime_substr in ['%Y','%m','%d','%j']):
        temp_datetime = datetime.strptime(datetime_fullstr.split(' ')[0], '%Y%j')
        substr = temp_datetime.strftime(datetime_substr)
    elif (datetime_substr in ['%H%M']):
        temp_datetime = datetime.strptime(datetime_fullstr.split(' ')[1], '%H%M')
        substr = temp_datetime.strftime(datetime_substr)

    return substr

def get_time(smoke_plume_df):
    """
    Get the time for the given times and views. We want to use the last time stamped photo but not if it is the next day
    Args:
        smoke_plume_df: smoke plume df with the shapes semi processed with the 'format_plume_data' function below

    Returns: series of times

    """
    time = np.where(
        np.logical_and((smoke_plume_df['start_doy'] != smoke_plume_df['end_doy']), (smoke_plume_df['view'] == 'C')),
        '2357',
        np.where(
            np.logical_and((smoke_plume_df['start_doy'] != smoke_plume_df['end_doy']), (smoke_plume_df['view'] == 'F')),
            '2345', smoke_plume_df['end_time']))

    return time

def get_doy(smoke_plume_df):
    """
    Get the doy for the given times and views. We want to use the last time stamped photo but not if it is the next day
    Args:
        smoke_plume_df: smoke plume df with the shapes semi processed with the 'format_plume_data' function below

    Returns: series of doy

    """
    doy = np.where(smoke_plume_df['start_doy'] != smoke_plume_df['end_doy'], smoke_plume_df['start_doy'], smoke_plume_df['end_doy'])

    return doy

def get_nearest_goes_minute(smoke_plume_df, start_min=2):
    """ 
    Find the nearest minute based on endtime for CONUS image
    Args:
        smoke_plume_df: smoke plume df with the shapes semi processed with the 'format_plume_data' function below
        start_min: 1 or 2 (April 3, 2019 (day 93) GOES switched schedule by 1 minute (previously started at :02 now :01))
    """
    # April 3, 2019 (day 93) GOES switched schedule by 1 minute (previously started at :02 now :01)
    goes_minutes = np.array([num for num in np.arange(start_min, 60, 5)]) 
    
    # search for nearest minute where CONUS image was taken (CONUS IS MUCH SMALLER SIZE VS FULL)
    goes_minutes_idx = np.searchsorted(goes_minutes, smoke_plume_df['end_time'].str.slice(2,4).to_numpy().astype(int))
    
    # for minutes exceeding the range, choose the next previous minute index (should be okay because
    # images are taken frequent enough that not much would have changed in a minute or two?
    goes_minutes_idx[(goes_minutes_idx==12)] = 11
    
    # the resulting end_time becomes the time we use to search for images
    return(smoke_plume_df['end_time'].str.slice(0,2) + np.char.zfill(goes_minutes[goes_minutes_idx].astype(str), 2))

## CURRENTLY JUST KEEP THE CONUS AND FULL DISK IMAGES ##
def format_plume_data(smoke_plume_df):
    """
    Function for cleaning up smoke plume data and creating needed columns
    Args:
        smoke_plume_df:

    Returns:

    """
    # create columns with date information
    smoke_plume_df['year'] = smoke_plume_df['Start'].apply(lambda x: year_month_day(x, '%Y'))
    smoke_plume_df['month'] = smoke_plume_df['Start'].apply(lambda x: year_month_day(x, '%m'))
    smoke_plume_df['day'] = smoke_plume_df['Start'].apply(lambda x: year_month_day(x, '%d'))
    smoke_plume_df['start_doy'] = smoke_plume_df['Start'].apply(lambda x: year_month_day(x, '%j'))
    smoke_plume_df['end_doy'] = smoke_plume_df['End'].apply(lambda x: year_month_day(x, '%j'))
    smoke_plume_df['start_time'] = smoke_plume_df['Start'].apply(lambda x: year_month_day(x, '%H%M'))
    smoke_plume_df['end_time'] = smoke_plume_df['End'].apply(lambda x: year_month_day(x, '%H%M'))

    # remove non CONUS or FD images (filter for min time stamps of just CONUS and FD) 
    # add in post april 2, 2019 scan changes
    conus_min = [str(num).zfill(2) for num in np.arange(2, 60, 5)] + [str(num).zfill(2) for num in np.arange(1, 60, 5)]
    fd_min = [str(num).zfill(2) for num in np.arange(0, 60, 15)] + [str(num).zfill(2) for num in np.arange(0, 60, 10)]
    smoke_plume_df['view'] = np.where(smoke_plume_df['start_time'].str.slice(2, 4).isin(conus_min), 'C',
                              np.where(smoke_plume_df['start_time'].str.slice(2, 4).isin(fd_min), 'F', 'other'))
        
    smoke_plume_df = smoke_plume_df[~(smoke_plume_df['view'] == 'other')].reset_index(drop=True)

    # create time and doy to use to search aws
    smoke_plume_df['doy'] = get_doy(smoke_plume_df)
    smoke_plume_df['time'] = get_time(smoke_plume_df)
    
    # find the nearest minute of CONUS image so that we can just use CONUS instead of FD
    smoke_plume_df['conus_time'] = np.where(((smoke_plume_df['year'].astype(int) >= 2019) & (smoke_plume_df['doy'].astype(int) >= 93)) | 
             (smoke_plume_df['year'].astype(int) >= 2020), 
             get_nearest_goes_minute(smoke_plume_df, start_min=1),
             get_nearest_goes_minute(smoke_plume_df, start_min=2))

    return smoke_plume_df

def generate_satpy_nc_tiff(bounds, save_nc_path, width=1200, height=1200, base_dir='../data/temp_netcdf/', desc='',
                           proj_desc='Geodetic Projection', datasets=['true_color', 'C07', 'C11']):
    """
    Process .nc files with satpy to generate true color and .tiff images
    Args:
        base_dir: temporary directory to store files
        bounds: bounds to help with resampling
        width: width of final image (4800 or 1200)
        height: height of final image (2700 or 1200)
        desc: description for satpy
        proj_desc: projection description for satpy
        datasets: satpy datasets to load and save (i.e. true color and different channels)

    Returns: .nc and .tiff files

    """
    # read files with satpy
    files = sp.find_files_and_readers(base_dir=base_dir, reader='abi_l1b')

    # create scene object
    scn = sp.Scene(reader='abi_l1b', filenames=files)
    scn.load(datasets)

    # resample (bring all channels to same resolution/ reproject)
    area_id = desc
    description = "{} in {}".format(desc, proj_desc)
    proj_id = desc
    proj_str = "epsg:4326"
    area_def = AreaDefinition(area_id, proj_id, description, proj_str, width, height, bounds)

    scn = scn.resample(area_def, resampler='nearest', cache_dir='../.satpy_temp')

    # save .tiff
    scn.save_datasets(writer='simple_image',
                      datasets=datasets,
                      filename='../data/img/{name}_' + h.get_satpy_filename(scn['true_color'].attrs) + '.png')

    # save .nc
    #scn.save_datasets(writer='cf', datasets=datasets, filename=save_nc_path+h.get_satpy_filename(scn['true_color'].attrs)+'.nc')

def img_crop_coordinates(img, output_size):
    """
    create random coordinates to crop images
    Args:
        img: image array
        output_size: output size of image

    Returns: coordinates used for cropping

    """
    w, h, c = img.shape
    th, tw = (output_size, output_size)
    if w == tw and h == th:
        return 0, 0
    
    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)

    return i, j

def img_crop(csv_df, output_size, num_crops, mask_dir="mask", crop_dir="crops", random_state=42):
    """ Create crops of the same height and width"""

    img_path = '../data/img/'
    map_path = f'../data/{mask_dir}/'

    # set random seed and sort dataframe so random order is replicable
    random.seed(a=random_state, version=2)

    for row in tqdm(csv_df.sort_values('true_color').itertuples()):

        # max tries counter
        counter = 0

        sat_img = io.imread(img_path + row.true_color)
        c07_img = io.imread(img_path + row.C07)
        c11_img = io.imread(img_path + row.C11)
        map_img = io.imread(map_path + row.mask)

        for num_crop in range(num_crops):

            i, j = img_crop_coordinates(sat_img, output_size)

            cropped_sat_img = sat_img[i:i + output_size, j:j + output_size]
            cropped_c07_img = c07_img[i:i + output_size, j:j + output_size]
            cropped_c11_img = c11_img[i:i + output_size, j:j + output_size]
            cropped_map_img = map_img[i:i + output_size, j:j + output_size]

            # have 60% of the crops have smoke in the image (deal with class imbalance)
            if (num_crop <= int(0.6 * num_crops)):
                # while (np.sum(cropped_map_img[:,:,:3]) / (output_size ** 2 * 255) < 0.03) and counter < 200:
                while ((np.max(cropped_map_img[:,:,:3])==0) & (counter < 200)):
                    i, j = img_crop_coordinates(sat_img, output_size)

                    cropped_sat_img = sat_img[i:i + output_size, j:j + output_size]
                    cropped_c07_img = c07_img[i:i + output_size, j:j + output_size]
                    cropped_c11_img = c11_img[i:i + output_size, j:j + output_size]
                    cropped_map_img = map_img[i:i + output_size, j:j + output_size]

                    counter += 1

            final_sat_img = Image.fromarray(cropped_sat_img[:,:,:3])
            final_c07_img = Image.fromarray(cropped_c07_img[:,:,:3])
            final_c11_img = Image.fromarray(cropped_c11_img[:,:,:3])
            final_map_img = Image.fromarray(cropped_map_img[:,:,:3])

            final_sat_img.save(f"../data/{crop_dir}/img/{row.true_color.split('.')[0]}_{num_crop}.png", compression=None)
            final_c07_img.save(f"../data/{crop_dir}/img/{row.C07.split('.')[0]}_{num_crop}.png", compression=None)
            final_c11_img.save(f"../data/{crop_dir}/img/{row.C11.split('.')[0]}_{num_crop}.png", compression=None)
            final_map_img.save(f"../data/{crop_dir}/mask/{row.mask.split('.')[0]}_{num_crop}.png", compression=None)
            
            
def raster_to_poly(batch_map, batch_fname, orig_bounds=(-124.48200299999999, 32.528832, -114.131211, 42.009502999999995), 
                   orig_width=1200, orig_height=1200, smooth_size=100):
    """
    batch_map
    batch_fname
    orig_bounds
    orig_width
    orig_height
    smooth_size
    """
    # get shapes
    N, C, W, H = batch_map.shape
    
    # set up transform for polygon coordinates
    out_west, out_south, out_east, out_north = orig_bounds
    transform = rasterio.transform.from_bounds(out_west, out_south, out_east, out_north, width=orig_width, height=orig_height)
    
    
    temp_dict = {'Start':[], 'End':[], 'geometry':[]}
    # iterate over the batch to get polygons
    # there has to be a faster way to do this....
    for idx in range(N):
        
        # use filename to get start and end strings to match HMS smoke data
        fname = batch_fname[idx]
        fname_split = os.path.basename(fname).split('/')[-1].split('_')
        start_str = fname_split[4]
        end_str = fname_split[5].split('.')[0]
        
        year = start_str[1:5]
        doy = fname_split[3][-3:]
        
        start_time = start_str[-4:]
        end_time = end_str[-4:]
        
        # extracted shapes as iter (sieve to smooth out rough edges...)
        shapes = rasterio.features.shapes(rasterio.features.sieve(batch_map[idx,0,:,:], smooth_size), transform=transform)
        for shapedict, value in shapes:
            
            # where mask is true save polygon
            if value == 1.0:

                temp_dict['Start'] += [f"{year}{doy} {start_time}"]
                temp_dict['End'] += [f"{year}{doy} {end_time}"]
                temp_dict['geometry'] += [shapely.geometry.shape(shapedict)]
    
    return temp_dict
        

## COCO ANNOTATION 
## for creating coco annotations to modify smoke polygons 

def _coco_row_annotation(row, transform):
    """
    generate the annotations for each smoke polygon
    Args:
        row: polygon row represented as a itertuple
        transform: inverse affine transform from geotiff metadata...convert from latlong to pixel coordinates

    Returns:
        annotation_dict: dictionary with the necessary components of a COCO json annotation

    """
    ## transform from lat long to pixel coordinates
    transformed_polygon_geom = shapely.affinity.affine_transform(row.geometry,
                                                                 [transform.a,
                                                                  transform.b,
                                                                  transform.d,
                                                                  transform.e,
                                                                  transform.xoff,
                                                                  transform.yoff])

    ## get bounds
    bounds = transformed_polygon_geom.bounds

    annotation_dict = {'id': '', # place holder for adding annotation id afterwards... cant keep track of # of plumes
                       'image_id': row.image_id,
                       'category_id': 1,
                       'segmentation': [h.polygon_to_coco(transformed_polygon_geom)],
                       'area': transformed_polygon_geom.area,
                       'bbox': [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]], # coco needs width and height
                       'iscrowd': 0}

    return(annotation_dict)

def coco_geom_annotation(df, transform):
    """
    convert shape polygons into annotation in the COCO format
    Args:
        df: filtered smoke dataframe
        transform: inverse affine transform from geotiff metadata...convert from latlong to pixel coordinates

    Returns:
        annotation_dict: dictionary with the necessary components of a COCO json annotation

    """
    ## convert geometry to json and store as list
    annotation_list = []
    for row in df.itertuples():
        annotation_list.append(_coco_row_annotation(row, transform))

    return(annotation_list)

def coco_img_dict(row, annotation_dict):
    """
    generate the dict that stores image information for COCO
    Args:
        row: polygon row represented as a itertuple
        annotation_dict: literally just to get the height and width... bad design.

    Returns:
        img_dict: dictionary with the necessary components of COCO image details

    """
    img_dict = {'height': annotation_dict['img_height'],
                'width': annotation_dict['img_width'],
                'file_name': row.true_color,
                'id': row.image_id,
                'date_captured': 0,
                'coco_url': '',
                'license': 0,
                'flickr_url': ''}

    return(img_dict)

def generate_coco_smoke_annotations(row, smoke_plume_df, img_path='../data/img/'):
    """
    Create json annotation file in the MS COCO format to be input into CVAT for polygon correction
    Args:
        row: row from index file with true color filename and image index
        smoke_plume_df: smoke plume dataframe
        img_path: path to where the input images are located

    Returns:
        dictionary with the filtered smoke dataframe (including additional columns) and annotation dict
    """
    ## open the geo .tiff file
    src = rasterio.open(img_path + row.true_color)

    ## get filter parameters from filename string
    param_dict = h.get_true_color_filename_split_dict(row.true_color)

    ## filter the smoke plume data to what we need
    filtered_smoke_df = h.filter_plumes(smoke_plume_df, param_dict)

    ## add image id
    filtered_smoke_df['image_id'] = row.image_id

    ## check to see if there are any multipolygons and if yes then explode multipolygons
    if filtered_smoke_df['geometry'].apply(lambda x: isinstance(x, shapely.geometry.multipolygon.MultiPolygon)).sum() > 0:

        ## keep the explode in case we want to recombine later...
        filtered_smoke_df = filtered_smoke_df.explode().reset_index(level=[1])
        filtered_smoke_df.rename(columns={'level_1': 'explode_id'}, inplace=True)
    else:
        filtered_smoke_df['explode_id'] = 0

    # ## add annotation id one for each smoke plume (even if multipolygon was split; add 1 cause COCO starts at 1)
    # filtered_smoke_df['annotation_id'] = np.arange(len(filtered_smoke_df)) + 1

    ## get inverse transformation to convert from lat long to pixels
    inverse_transform = ~src.meta['transform']

    ## convert to COCO style annotation
    annotation = coco_geom_annotation(filtered_smoke_df, inverse_transform)

    return({'new_smoke_df': [filtered_smoke_df],
            'annotation': annotation,
            'img_height': src.height,
            'img_width': src.width})

def generate_coco_json_smoke_df(index_df, smoke_plume_df, save_smoke_data_fn, save_json_data_fn):
    """
    Create json annotation file in the MS COCO format to be input into CVAT for polygon correction and
    new smoke dataframe with important columns for matching plumes to images
    Args:
        row: row from index file with true color filename and image index
        smoke_plume_df: smoke plume dataframe
        save_smoke_data_fn: filename to save smoke dataframe
        save_json_data_fn: filename to save coco json

    Returns:
        dictionary with the filtered smoke dataframe (including additional columns) and annotation dict
    """
    coco_json_dict = {}

    ## prep lists for storing info
    new_smoke_df_list = []
    annotation_list = []
    image_list = []

    ## iterate and generate json 
    # FIXME REMOVE .head() AFTER TESTING
    for row in index_df.head().itertuples():
        ## get the annotation dictionary for smoke plumes
        smoke_annotation_dict = generate_coco_smoke_annotations(row, smoke_plume_df)

        ## store the filtered dataframe and annotation list
        new_smoke_df_list += smoke_annotation_dict['new_smoke_df']
        annotation_list += smoke_annotation_dict['annotation']

        ## image information
        image_list.append(coco_img_dict(row, smoke_annotation_dict))

    ## combine list of dataframes
    new_smoke_df = pd.concat(new_smoke_df_list, sort=False).reset_index(drop=True)

    ## increment and create annotation id now that mutlipolygons have been exploded (add 1 because coco starts at 1)
    new_smoke_df['annotation_id'] = np.arange(start=1, stop=len(new_smoke_df) + 1)

    for idx in range(len(annotation_list)):
        annotation_list[idx]['id'] = int(new_smoke_df.loc[idx,'annotation_id'])

    ## add annotation and image data
    coco_json_dict['images'] = image_list
    coco_json_dict['annotations'] = annotation_list

    ## dont really need these things for now but just following the format...
    coco_json_dict['licenses'] = [{'id': 0, 'name': '', 'url': ''}]
    coco_json_dict['info'] = {'version': '', 'year': '', 'date_created': '',
                               'contributor': '', 'description': '', 'url': ''}
    coco_json_dict['categories'] = [{'id': 1, 'name': 'smoke', 'supercategory': ''}]

    ## save new smoke dataframe
    new_smoke_df.to_file(f"../data/{save_smoke_data_fn}.geojson", driver='GeoJSON')

    ## save coco json data
    with open(f"../data/{save_json_data_fn}.json", 'w') as outfile:
        json.dump(coco_json_dict, outfile)
