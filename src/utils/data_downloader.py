from tqdm import tqdm
from botocore.config import Config
import xarray as xr
import netCDF4
import requests
import boto3
import os
import glob
import numpy as np

from utils import helpers as h
from utils import data_prep as dp

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('../../logger.log', 'a'))
print = logger.info

def get_s3_keys(bucket, prefix = '', client=None):
    """Generate the keys in an S3 bucket. https://github.com/HamedAlemo/visualize-goes16/blob/master/visualize_GOES16_from_AWS.ipynb

    Arguments:
        bucket {str} -- Name of the S3 bucket.
    
    Keyword Arguments:
        prefix {str} -- Only fetch keys that start with this prefix (optional). (default: {''})
    """
    if client is not None:
        s3 = client
    else:
        s3 = boto3.client('s3')
    kwargs = {'Bucket': bucket}

    if isinstance(prefix, str):
        kwargs['Prefix'] = prefix

    while True:
        resp = s3.list_objects_v2(**kwargs)

        try:
            for obj in resp['Contents']:
                key = obj['Key']
                if key.startswith(prefix):
                    yield key

            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break
        except KeyError:
            break

def get_netcdf4(bucket_name, key):
    """
    Download netcdf4 data
    Args:
        bucket_name: S3 bucket name
        key: key to use for downloading from AWS

    Returns: xarray dataset

    """

    # get data from s3
    resp = requests.get('https://' + bucket_name + '.s3.amazonaws.com/' + key)

    # store data as xarray
    nc4_ds = netCDF4.Dataset('temp_name', mode='r', memory=resp.content)
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    data = xr.open_dataset(store)

    return data

def aws_goes_download_metpy(year, doy, hour, minute, band, view, extra_desc='', bounds=None, save_data_path=None,
                    bucket_name='noaa-goes16', product_name='ABI-L1b-Rad', satellite='G16', client=None):
    """function for downloading goes data from aws
    
    Arguments:
        year {int or str} -- year of goes data
        doy {int or str} -- day of year of goes data
        hour {int or str} -- hour of goes data
        minute {int or str} -- minute of goes data
        band {int or str} -- band of goes data
        view {str} -- view for goes data (e.g. C or F)
        extra_desc {str} -- description to append to filename (default: {''})
    
    Keyword Arguments:
        bounds {iterable list or tuple} -- bounds in latlong (xmin, ymin, xmax, ymax)
        save_data_path {str} -- path to save data (default: {None})
        bucket_name {str} -- name of the bucket (default: {'noaa-goes16'})
        product_name {str} -- noaa product name (default: {'ABI-L1b-RadC'})
        satellite {str} -- satellite data to pull from; must match aws filename specification (default: {'G16'})
        client {boto3 client} -- client for boto3
    
    Returns:
        [.nc] -- if no save_data_path provided, then return netcdf4 file
    """
    # search string for AWS
    search_str = "{0}{7}/{1}/{2}/{3}/OR_{0}{7}-M3C{4}_{5}_s{1}{2}{3}{6}".format(product_name,
                                                         year,
                                                         str(doy).zfill(3),
                                                         str(hour).zfill(2),
                                                         str(band).zfill(2),
                                                         str(satellite),
                                                         str(minute).zfill(2),
                                                         view)

    if client is not None:
        # get keys associated with the parameters provided 
        keys = get_s3_keys(bucket_name, prefix=search_str, client=client)
    else:
        # get keys associated with the parameters provided 
        keys = get_s3_keys(bucket_name, prefix=search_str)

    # just get the first file given the hour and minute (there should only be one that matches)
    try:
        key = [key for key in keys][0] 

        # get the netcdf4 data from AWS
        data = get_netcdf4(bucket_name, key)

        proj = data.metpy.parse_cf('Rad')

        # add the max radiance value for visualization correction later
        proj.attrs['max_rad'] = data.max_radiance_value_of_valid_pixels.values

        # store raw projection information (because xarray cant serialize projection object...)
        # need to jump over some hurdles because when metpy creates proj it changes units to meters
        proj_raw_info = data['goes_imager_projection'].copy()
        proj_raw_info.coords['y_image'] = proj.coords['y_image']
        proj_raw_info.coords['x_image'] = proj.coords['x_image']

        # convert to reflectance
        # proj = h.get_reflectance(proj, data.esun)

        if bounds is not None:
            proj = h.get_slice(proj, bounds)

        # save data if file path is provided
        if save_data_path is not None:

            # drop crs so that we can save using xarray function
            proj_raw_info = proj_raw_info.drop('crs')
            proj = proj.drop('crs')

            # get filename
            file_name = h.get_filename(year, doy, hour, minute, satellite, band, extra_desc, view)

            # create dataset to save for xarray
            proj_dataset = xr.Dataset({'Rad': proj.copy(deep=True), 'goes_imager_projection': proj_raw_info})
            proj_dataset.to_netcdf(save_data_path + file_name, encoding={'x': {'dtype': 'float32'}, 'y': {'dtype': 'float32'}})
        
        else:
            return proj

    except IndexError:
        print("No AWS file matching: {}".format(search_str))

def goes_download_wrapper_metpy(smoke_plume_data, save_data_path, extra_desc, bounds, band):
    """Wrapper that makes use of the aws_goes_downloader function to save aws data 

    Arguments:
        smoke_plume_data {pandas dataframe} -- smoke plume data with details for goes data to retrieve (year, doy, hour, etc.)
        save_data_path {str} -- path to save data (default: {None})
        extra_desc {str} -- description to append to filename (default: {''})
        bounds {iterable list or tuple} -- bounds in latlong (xmin, ymin, xmax, ymax)
        band {int or str} -- band of goes data
    """
    curr_dir_files = [file for file in os.listdir(save_data_path) if file[0] != '.']
    
    # create s3 client
    config = Config(retries = dict(max_attempts = 20))
    s3 = boto3.client('s3', config=config, region_name='us-east-1')

    for row in tqdm(smoke_plume_data.itertuples()):
        
        # get filenames for the current files already in the directory
        file_name = h.get_filename(year=row.year, doy=row.doy, hour=row.time[0:2], minute=row.time[2:4],
                                satellite=row.Satellite, band=band, extra_desc=extra_desc, view=row.view)

        if file_name not in curr_dir_files:

            aws_goes_download_metpy(year=row.year, doy=row.doy, hour=row.time[0:2], minute=row.time[2:4], band=band,
                            view=row.view, extra_desc=extra_desc, bounds=bounds, save_data_path=save_data_path, client=s3)

            curr_dir_files.append(file_name)

def aws_goes_download_satpy(year, doy, hour, minute, band, view='C', extra_desc='', bounds=None, save_data_path=None,
                            bucket_name='noaa-goes16', product_name='ABI-L1b-Rad', satellite='G16', client=None):
    """ Main function to download data from aws
    Args:
        year: year of goes data
        doy: day of year of goes data
        hour: hour of goes data
        minute: minute of goes data
        band: band of goes data
        view: view for goes data (e.g. C or F)
        extra_desc: description to append to filename (default: {''}) NOT CURRENTLY IN USE...
        bounds: bounds in latlong (xmin, ymin, xmax, ymax)
        save_data_path: path to save data (default: {None})
        bucket_name: name of the bucket (default: {'noaa-goes16'})
        product_name: noaa product name (default: {'ABI-L1b-RadC'})
        satellite: satellite data to pull from; must match aws filename specification (default: {'G16'})
        client: client for boto3
    """

    # search string for AWS april 2, 2019 goes default mode changed to M6 scanning
    #search_str = "{0}{7}/{1}/{2}/{3}/OR_{0}{7}-M3C{4}_{5}_s{1}{2}{3}{6}".format(product_name,
    search_str = "{0}{7}/{1}/{2}/{3}/OR_{0}{7}-M6C{4}_{5}_s{1}{2}{3}{6}".format(product_name,
                                                                            year,
                                                                                str(doy).zfill(3),
                                                                                str(hour).zfill(2),
                                                                                str(band).zfill(2),
                                                                                str(satellite),
                                                                                str(minute).zfill(2),
                                                                                view)

    if client is not None:
        # get keys associated with the parameters provided
        keys = get_s3_keys(bucket_name, prefix=search_str, client=client)
    else:
        # get keys associated with the parameters provided
        keys = get_s3_keys(bucket_name, prefix=search_str)

    # just get the first file given the hour and minute (there should only be one that matches)
    try:
        key = [key for key in keys][0]

        # get the netcdf4 data from AWS
        data = get_netcdf4(bucket_name, key)

        if bounds is not None:
            data = h.get_slice_latlng(data, bounds)

        # save data if file path is provided
        if save_data_path is not None:

            data.to_netcdf(save_data_path + data.dataset_name, encoding={'x': {'dtype': 'float32'}, 'y': {'dtype': 'float32'}})

        else:
            return data

    except IndexError:
        print("No AWS file matching: {}".format(search_str))


def goes_download_wrapper_satpy(smoke_plume_data, temp_data_path, save_data_path, extra_desc, bounds, bands, curr_dir_files=None):
    """Wrapper that makes use of the aws_goes_downloader function to save aws data

    Arguments:
        smoke_plume_data {pandas dataframe} -- smoke plume data with details for goes data to retrieve (year, doy, hour, etc.)
        temp_data_path {str} -- path to save data temporarily before satpy processing (default: {None})
        extra_desc {str} -- description to append to filename (default: {''})
        bounds {iterable list or tuple} -- bounds in latlong (xmin, ymin, xmax, ymax)
        bands {list} -- bands of goes data
        curr_dir_files {list} -- list of save data path directory files
    """
    if curr_dir_files is None:
        curr_dir_files = [file[11:-5] for file in os.listdir('../data/img/') if (file[0:4] == 'true')]

    # create s3 client
    config = Config(retries=dict(max_attempts=20))
    s3 = boto3.client('s3', config=config, region_name='us-east-1')

    for row in tqdm(smoke_plume_data.itertuples()):
        
        # set file name
        file_name = '{0}_doy{5}_s{1}{2}{3}{4}_e{1}{2}{3}'.format('G16', row.year, str(row.month).zfill(2),
                                                          str(row.day).zfill(2),
                                                          row.conus_time, str(row.doy).zfill(3))

        # get filenames for the current files already in the directory
        file_seen = any(file_name in files for files in curr_dir_files)

        # only try to download the files that have not been downloaded yet
        if file_seen:
            print("skipping {} | doy: {} | conus_time: {}".format(file_name,str(row.doy), str(row.conus_time)))
            continue

        else:
            print("doy: {} | conus_time: {}".format(str(row.doy), str(row.conus_time)))
            for band in bands:

                # download goes imagery, crop, and save in temporary directory
                aws_goes_download_satpy(year=row.year, doy=row.doy, hour=row.conus_time[0:2], minute=row.conus_time[2:4], 
                                        band=band, view="C", extra_desc=extra_desc, bounds=bounds, 
                                        save_data_path=temp_data_path, client=s3)

            try:
                # process using satpy to generate true color image
                dp.generate_satpy_nc_tiff(bounds=bounds, save_nc_path=save_data_path, base_dir=temp_data_path,
                           #width=4800, height=2700, desc='',
                           width=1200, height=1200, desc='',
                           proj_desc='Geodetic Projection', datasets=['true_color', 'C07', 'C11'])

            except ValueError:
                print('No files for SatPy to open')

            except KeyError:
                print('Missing files; moving to next file')

        curr_dir_files.append(file_name)

        # delete original .nc files
        for file in glob.glob(temp_data_path+'*.nc'):
            os.remove(file)
