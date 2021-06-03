from joblib import Parallel, delayed, cpu_count
from utils import data_prep as dp
from tqdm import tqdm

import geopandas as gpd
import pickle
import argparse

## num cpu to use
NUM_CPU = cpu_count() - 1

def raster_to_poly_wrapper(i, out_map_store_dict):
    smoke_dict_list = []

    fname = out_map_store_dict['fname'][i]
    out_maps = out_map_store_dict['out_map'][i]

    # get polygons from predicted rasters
    temp_dict = dp.raster_to_poly(out_maps, fname)
    smoke_dict_list.append(temp_dict)
        
    return(smoke_dict_list)


def main(pickle_path, out_path):
    
    # load pickled file
    with open(pickle_path, 'rb') as f:
        out_map_store_dict = pickle.load(f)
        
    # get the smoke polygons
#     smoke_dict_list = Parallel(n_jobs=NUM_CPU)(delayed(raster_to_poly_wrapper)(i=i, out_map_store_dict=out_map_store_dict)\
#                                                for i in range(len(out_map_store_dict['fname'])))


    smoke_dict_list = []
    for i in tqdm(range(len(out_map_store_dict['fname'])), desc='raster_to_poly'):
        smoke_dict_list += raster_to_poly_wrapper(i, out_map_store_dict)

    # generate one smoke dict for whole dataset
    smoke_dict = {'Start':[], 'End':[], 'geometry':[]}      

    for temp_dict_element in smoke_dict_list:
        
        # joblib returns list
        temp_dict = temp_dict_element
        
        smoke_dict['Start'] += temp_dict['Start']
        smoke_dict['End'] += temp_dict['End']
        smoke_dict['geometry'] += temp_dict['geometry']

    # get geodataframe
    smoke_df = gpd.GeoDataFrame(smoke_dict, crs="EPSG:4326")

    if smoke_df.shape[0] != 0:

        # format plumes by creating intermediate features
        smoke_df = dp.format_plume_data(smoke_df)

        try: 
            # save output as geojson of smoke polygons
            smoke_df.to_file(out_path, driver='GeoJSON')
        except ValueError as e:
            print(f"No smoke plumes to write with error: {e}")

    else:
        print(f"No smoke plumes to write!")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raster to polygon')
    parser.add_argument('pickle_file', metavar='PATH',
                        help='path to pickled model output with filenames')
    parser.add_argument('--out_file', type=str, metavar='PATH',
                        help='path to output file')
    
    args = parser.parse_args()
    
    main(args.pickle_file, args.out_file)