import requests.exceptions as re
import warnings
import geopandas as gpd
import pandas as pd
import time

from utils import data_prep as dp
from utils import data_downloader as dd
from utils import helpers as h

warnings.simplefilter("ignore", (UserWarning, FutureWarning, RuntimeWarning))

if __name__ == "__main__":

    ## file paths
    DATA_FILE_PATH = '../data/'
    NETCDF_FILE_PATH = DATA_FILE_PATH + 'netcdf/'
    TEMP_NETCDF_FILE_PATH = DATA_FILE_PATH + 'temp_netcdf/'

    ## read in plume data
    #plumes_df = gpd.read_file(DATA_FILE_PATH + "smoke_plumes/us_plumes_2018-2020.geojson")
    #plumes_df = gpd.read_file(DATA_FILE_PATH + "smoke_plumes/ca-nv_plumes_2019-2020.geojson")
    plumes_df = pd.read_csv(DATA_FILE_PATH + "smoke_plumes/ca-nv_test_2019-2020.csv", dtype = str)

    ## temp plume data
    #temp_plumes_df = plumes_df[(plumes_df['conus_time'].str.slice(0,2)>='12') & (plumes_df['conus_time'].str.slice(0,2)<='23') & (plumes_df['conus_time'].str.slice(2,4).isin(['02','01','31','32']))]
    #temp_plumes_df = plumes_df[(plumes_df['conus_time'].str.slice(0,2)>='12') & (plumes_df['conus_time'].str.slice(0,2)<='23') & (plumes_df['month'] > '04')]
    temp_plumes_df = plumes_df
    

    start_time = time.time()
    print("Starting download")

    while True:
        try:
            dd.goes_download_wrapper_satpy(smoke_plume_data=temp_plumes_df,
                                           temp_data_path=TEMP_NETCDF_FILE_PATH,
                                     save_data_path=NETCDF_FILE_PATH, 
                                     extra_desc='', 
                                     #bounds=(-124.5, 24.4, -66.6, 49.3), # US bounds
                                     bounds=(-124.48200299999999, 32.528832, -114.131211, 42.009502999999995), # CA/NV bounds
                                     bands=[1,2,3,7,11])
            break
        except (re.SSLError, re.ConnectionError, re.ChunkedEncodingError) as e:
            print("Connection Error. Continuing.")
            
    print("--- %s seconds ---" % (time.time() - start_time))
