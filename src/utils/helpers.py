import os
import numpy as np
import geopandas as gpd
import pandas as pd
# import cartopy.crs as ccrs
# import metpy
import glob
from utils import errors as e
from sklearn.model_selection import train_test_split
from shapely.geometry import MultiPolygon, Polygon

def poly_intersection(row, group_df, binary_mode=False):
    """Identify the polygons that have overlaps with other polygons

    Arguments:
        row {pandas row} -- Passed in from an apply function
        group_df {pandas dataframe} -- Filtered dataframe of the matching day of year

    Keyword Arguments:
        binary_mode {bool} -- Option for returning binary mode vs string of intersections (default: {False})

    Returns:
        [bool or str] -- Either boolean or string depending on binary mode option
    """
    if binary_mode:
        intersections = any([row[4].intersects(row_2[5]) for row_2 in group_df.itertuples() if row.name != row_2[0]])
    else:
        intersections = '-'.join(sorted(set([str(row_2[0]) for row_2 in group_df.itertuples() if
                                             (row.name == row_2[0]) | (row[4].intersects(row_2[5]))])))
    return intersections

def get_slice(proj, bounds, crs=None):
    """Subset the original data and return a bounded version (ONLY GOT METPY PROCESSED DATAARRAYS)

    Arguments:
        proj {netcdf4 xarray} -- xarray of netcdf4 data
        bounds {iterable list or tuple} -- bounds in latlong (xmin, ymin, xmax, ymax)
    """
    if 'crs' in proj.coords:
        # convert from lat long to radians
        rad_bounds = proj.metpy.cartopy_crs.transform_points(ccrs.PlateCarree(),
                                                             np.array([bounds[0], bounds[2]]),
                                                             np.array([bounds[1], bounds[3]]))
    elif crs is not None:
        # convert from lat long to radians
        rad_bounds = crs.transform_points(ccrs.PlateCarree(),
                                          np.array([bounds[0], bounds[2]]),
                                          np.array([bounds[1], bounds[3]]))

    # selecting a subset of the data
    # need to have reversed y indices because labels for the Y in the data are monotonically decreasing
    # XArray requires the label order to match the direction of the labels
    new_proj = proj.sel(x=slice(rad_bounds[0][0], rad_bounds[1][0]),
                        y=slice(rad_bounds[1][1], rad_bounds[0][1]))


    return new_proj

def get_slice_latlng(ds, latlng_bounds):
    """
    Get slice given given lat long bounds
    Args:
        ds: original dataset
        latlng_bounds: bounds in latlong (xmin, ymin, xmax, ymax)

    Returns: sliced dataset

    """

    rad_bounds = latlng_to_rad_bbox(latlng_bounds, ds.goes_imager_projection)

    new_ds = ds.sel(x=slice(rad_bounds[0], rad_bounds[2]),
                        y=slice(rad_bounds[3], rad_bounds[1]))

    return new_ds

def get_reflectance(proj, esun):
    """
    Convert from radiances to reflectance
    Args:
        proj: xarray data array of the radiance projection from GOES
        esun: conversion factor (should be found in netcdf4 metadata

    Returns:
        xarray of reflectances
    """
    ref = (proj * np.pi * 0.3) / esun
    ref = np.maximum(ref, 0.0)
    ref = np.minimum(ref, 1.0)

    return ref

def get_filename(year, doy, hour, minute, satellite, band, extra_desc, view, orig_dataset_name=False):
    """Generate file name to saving to disk

    Arguments:
        year {int or str} -- year of goes data
        doy {int or str} -- day of year of goes data
        hour {int or str} -- hour of goes data
        minute {int or str} -- minute of goes data
        satellite {str} -- satellite data to pull from; must match aws filename specification (default: {'G16'})
        band {int or str} -- band of goes data
        extra_desc {str} -- description to append to filename (default: {''})
        view {str} -- satellite view (e.g. C or F)
        orig_dataset_name {bool} -- true if names in folder are original dataset names
    """
    if satellite in ['GOES-EAST', 'GOES-WEST']:
        sat_dict = {'GOES-EAST': 'G16', 'GOES-WEST': 'G17'}
        satellite = sat_dict[satellite]

    if not orig_dataset_name:
        # create file name to save file
        file_name = "{0}{1}{2}{3}_{7}_{4}_C{5}_{6}.nc".format(year,
                                                              str(doy).zfill(3),
                                                              str(hour).zfill(2),
                                                              str(minute).zfill(2),
                                                              str(satellite),
                                                              str(band).zfill(2),
                                                              extra_desc,
                                                              view)
    else:
        # create file name to save .nc files (must be original name)
        file_name = 'OR_ABI-L1b-Rad{0}-M3C{1}_{2}_s{3}{4}{5}{6}'.format(view,
                                                    str(band).zfill(2),
                                                    satellite,
                                                    year,
                                                    doy,
                                                    str(hour).zfill(2),
                                                    str(minute).zfill(2))





    return (file_name)

def get_satpy_filename(scn_attrs):
    """
    Create filename given the scene object attributes
    Args:
        scn_attrs: scene object attributes generated with satpy

    Returns:
        formatted string with filename
    """
    sdt = scn_attrs['start_time']
    edt = scn_attrs['end_time']
    start_hr = str(sdt.hour).zfill(2)
    end_hr = str(edt.hour).zfill(2)
    start_min = str(sdt.minute).zfill(2)
    end_min = str(edt.minute).zfill(2)
    start_day = str(sdt.day).zfill(2)
    end_day = str(edt.day).zfill(2)
    year = str(sdt.year)
    month = str(sdt.month).zfill(2)
    doy = sdt.strftime('%j').zfill(3)
    satellite = scn_attrs['platform_shortname']

    return('{0}_doy{9}_s{1}{2}{3}{4}{5}_e{1}{2}{6}{7}{8}'.format(satellite, year, month,
                                                         start_day, start_hr, start_min,
                                                         end_day, end_hr, end_min, doy))

def get_true_color_filename_slug(row):
    """
    Get filenames from the final smoke plume dataframe (to filter all images to the ones with matching plumes)
    Args:
        row: geopandas row tuple from itertuple

    Returns: series

    """
    doy = row.doy
    year = row.year
    month = row.month
    day = row.day
    time = row.time

    file_name_str = 'true_color_G16_doy{0}_s{1}{2}{3}{4}'.format(str(doy).zfill(3),
                                                                 str(year).zfill(4),
                                                                 str(month).zfill(2),
                                                                 str(day).zfill(2),
                                                                 str(time).zfill(4))

    return(file_name_str)

def get_true_color_filename_split_dict(fname):
    """
    Get parameters to filter smoke dataframe using true color .tiff file name information
    Args:
        fname: string representing the true_color filename

    Returns:
        param_dict: dictionary with necessary parameters for filtering

    """
    # split string to get smoke plume filter parameters
    split_string = fname.split('/')[-1].split('_')[4]
    year = split_string[1:5]
    month = split_string[5:7]
    day = split_string[7:9]
    time = split_string[-4:]

    return({'year':year, 'month':month, 'day':day, 'time':time})

def get_plume_df(smoke_data_path, drop_null=True):
    """
    Get dataframe of smoke plume data (with shape polygons)
    Args:
        smoke_data_path: file path to the smoke plume data
        drop_null: flag to drop null observations

    Returns:
        dataframe with smoke plume data
    """
    # read data
    smoke_file_names = sorted(
        [file_name for file_name in os.listdir(smoke_data_path) if file_name.split('.')[-1] == 'shp'])

    # read in multiple shape files and concatenate into one dataframe
    smoke_plume_gpd = [gpd.read_file(smoke_data_path + file) for file in smoke_file_names]

    result = gpd.GeoDataFrame(pd.concat(smoke_plume_gpd, ignore_index=True, sort=True),
                              crs=gpd.read_file(smoke_data_path + smoke_file_names[0]).crs)

    if drop_null:
        # remove observations that are null
        result = result[~result['Start'].isnull()].reset_index(drop=True)

    return result

def get_state_shapes(state_shape_file, states_list=None):
    """
    Filter the US state shape file into states that we are about
    Args:
        state_shape_file: US state shapes file
        states_list: list of state names

    Returns:
        dataframe of state shapes
    """
    # read in shape files for states
    states_df = gpd.read_file(state_shape_file)

    if states_list is not None:
        states_df = states_df[states_df['NAME'].isin(states_list)]

    states_df = states_df.sort_values(by=['NAME'])

    return states_df

def filter_state_plumes(plume_df, states, state_shape_file='../data/states/tl_2017_us_state.shp'):
    """
    Filter to the plume shapes for the given states
    Args:
        plume_df: geopandas dataframe of the smoke plume data
        states: single string or list of strings of state names
        state_shape_file: shape file for US states

    Returns:
        dataframe with the plumes that were located within the states provided
    """

    if states is not None:
        if isinstance(states, str):
            states = [states]
        elif not isinstance(states, list):
            raise e.UnsupportedType('States expected to be single str or list of str')

    states = sorted(states)

    # get the state shapes into a dataframe
    state_shapes_df = get_state_shapes(state_shape_file, states_list=states)

    # first create a series of bool for the first state
    filter_series = plume_df['geometry'].within(state_shapes_df['geometry'].iloc[0])

    for state_row in state_shapes_df.iloc[1:].itertuples():
        filter_series = filter_series | plume_df['geometry'].within(state_row.geometry)

    return plume_df[filter_series].reset_index(drop=True)

def filter_plumes(plume_df, param_dict):
    """
    Filter to the plume shapes given filename parameter dictionary. Also includes a 3 hour range given the way
    annotations are drawn.
    Args:
        plume_df: geopandas dataframe of the smoke plume data
        param_dict: dictionary with parameters from .tiff file name such as year, doy, time, etc. used to filter smoke

    Returns:
        dataframe with the plumes that match the criteria
    """

    # we want to keep the smoke plumes that are before the time stamp of the image, but after the night (in UTC)
    filtered_plume_df = plume_df[(plume_df['year'] == param_dict['year']) &
                              (plume_df['month'] == param_dict['month']) &
                              (plume_df['day'] == param_dict['day']) &
                              (plume_df['time'].astype(int) <= int(param_dict['time'])) &
                              (plume_df['time'].astype(int) > int(param_dict['time']) - 180)]

    return(filtered_plume_df.copy())

def latlng_to_rad(lat, lng, goes_proj_details=None):
    """ Function for converting lat long to radians 
    
    Args:
        lat: GRS80 geodetic latitude
        lng: GRS80 longitude
        goes_proj_details: goes projection details from xarray data array ('goes_imager_projection')

    Returns: tuple of radians for x and y

    """
    phi = (lat * np.pi) / 180.0
    lamb = (lng * np.pi) / 180.0

    if goes_proj_details is not None:
        perspective_point_height = goes_proj_details.attrs['perspective_point_height']
        r_eq = goes_proj_details.attrs['semi_major_axis']
        r_pol = goes_proj_details.attrs['semi_minor_axis']
        lambda_0 = (goes_proj_details.attrs['longitude_of_projection_origin'] * np.pi) / 180.0
        H = perspective_point_height + r_eq
        e = np.sqrt((np.power(r_eq, 2.0)-np.power(r_pol, 2.0))/np.power(r_eq, 2.0))
        phi_c = np.arctan(np.power(r_pol, 2.0) / np.power(r_eq, 2.0) * np.tan(phi))

        r_c = r_pol / np.sqrt(1 - np.power(e, 2.0) * np.cos(phi_c))
        s_x = H - r_c * np.cos(phi_c) * np.cos(lamb - lambda_0)
        s_y = -r_c * np.cos(phi_c) * np.sin(lamb - lambda_0)
        s_z = r_c * np.sin(phi_c)

        rad_y = np.arctan(s_z / s_x)
        rad_x = np.arcsin(-s_y / np.sqrt(np.power(s_x, 2.0) + np.power(s_y, 2.0) + np.power(s_z, 2.0)))

    else:

        perspective_point_height = 35786023.0
        r_eq = 6378137.0  # semi_major_axis
        r_pol = 6356752.31414  # semi_minor_axis
        lambda_0 = (-75.0 * np.pi) / 180.0  # longitude_of_projection_origin
        H = perspective_point_height + r_eq
        e = 0.0818191910435
        phi_c = np.arctan(np.power(r_pol, 2.0) / np.power(r_eq, 2.0) * np.tan(phi))

        r_c = r_pol / np.sqrt(1 - np.power(e, 2.0) * np.cos(phi_c))
        s_x = H - r_c * np.cos(phi_c) * np.cos(lamb - lambda_0)
        s_y = -r_c * np.cos(phi_c) * np.sin(lamb - lambda_0)
        s_z = r_c * np.sin(phi_c)

        rad_y = np.arctan(s_z / s_x)
        rad_x = np.arcsin(-s_y / np.sqrt(np.power(s_x, 2.0) + np.power(s_y, 2.0) + np.power(s_z, 2.0)))

    return rad_x, rad_y

def latlng_to_rad_bbox(bounds, goes_proj_details=None):
    """
    Convert from lat long to radian bounding box coordinates
    Args:
        bounds: bounds in latlong (xmin, ymin, xmax, ymax)
        goes_proj_details: ('goes_imager_projection') from xarray data array

    Returns:

    """
    if goes_proj_details is not None:
        x1, y1 = latlng_to_rad(bounds[1], bounds[0], goes_proj_details)
        x2, y2 = latlng_to_rad(bounds[3], bounds[2], goes_proj_details)
    else:
        x1, y1 = latlng_to_rad(bounds[1], bounds[0])
        x2, y2 = latlng_to_rad(bounds[3], bounds[2])

    return (x1, y1, x2, y2)

def create_train_test(idx_list, train_perc, random_state=42):
    """
    Create split list
    Args:
        idx_list: must be a list of idx integers
        train_perc: training percentage from dataset

    Returns: list with 'train', 'test' values

    """
    idx_split = train_test_split(idx_list,
                                 random_state=random_state,
                                 train_size=train_perc)

    train_test_list = np.array(['train'] * len(idx_list))
    train_test_list[idx_split[1]] = 'test'

    return (train_test_list)

def create_train_val_test(idx_list, train_perc, random_state=42):
    """
    Create split list
    Args:
        idx_list: list of img slug names
        train_perc: training percentage from dataset

    Returns: list with 'train', 'valid', 'test' values

    """
    # split train and test
    train_idx_split, test_idx_split = train_test_split(idx_list,
                                 random_state=random_state,
                                 train_size=train_perc)
    
    # split val off of test (50% for validation)
    val_idx_split, test_idx_split = train_test_split(test_idx_split,
                             random_state=random_state,
                             train_size=0.5)

    return train_idx_split, val_idx_split, test_idx_split

## adapted from https://solaris.readthedocs.io/en/latest/api/utils.html#solaris.utils.geo.polygon_to_coco
def polygon_to_coco(polygon):
    """Convert a geometry to COCO polygon format."""
    if isinstance(polygon, Polygon):
        coords = polygon.exterior.coords.xy
    elif isinstance(polygon, str):  # assume it's WKT
        coords = loads(polygon).exterior.coords.xy
    else:
        raise ValueError('polygon must be a shapely geometry or WKT.')
    # zip together x,y pairs
    coords = list(zip(coords[0], coords[1]))
    coords = [item for coordinate in coords for item in coordinate]

    return coords

## logger for training
def logger(info_str, log_fn_slug="../training_logs/training_log"):
    ## write to log file
    log = open(log_fn_slug+'.txt', "a+")  # append mode and create file if it doesnt exist
    log.write(info_str +
              "\n")
    log.close()