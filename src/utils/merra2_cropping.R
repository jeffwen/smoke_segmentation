library(ncdf4)
library(raster)
library(sf)
library(terra)
library(dplyr)
library(tidyr)

## this is where merra2 files are stored...
setwd("~/Stanford/projects/camp_fire/")

## get netcdf4 file names to read in
data_files <- list.files('data/merra2', full.names=T)

## sort files based on date so its in the right order
## split filename string and just take the date part
data_files_df <- data.frame(data_files) 
data_files_df <- data_files_df %>% 
    tidyr::separate(data_files, into=c(NA,NA,"date"), sep="\\.", extra='drop', remove=FALSE) %>% 
    arrange(date) %>%
    filter(((date>="20190501" & date<="20191031") | (date>="20200501" & date<="20201031")))

## terra open raster stack and rename to dates
raw_merra2_stack <- terra::rast(data_files_df$data_files)
names(raw_merra2_stack) <- data_files_df$date

## crop to CA & NV
out_shape <- terra::rast(nrow=1200, ncol=1200)
terra::ext(out_shape) <- terra::ext(-124.48200299999999, -114.131211, 32.528832, 42.009502999999995)
terra::crs(out_shape) <- "epsg:4326"

## resample to larger 1200x1200 shape
merra2_stack <- terra::resample(raw_merra2_stack, out_shape, 
                                     method='bilinear', wopt=list(progress=1))

## write as .tiff for each day of data
terra::writeRaster(merra2_stack, 
                   filename=paste(paste("data/merra2_tiff/merra2",
                                        stringr::str_sub(names(merra2_stack),1,-1),
                                        sep="_"),".tiff", sep=""),
                   wopt=list(datatype='FLT4S', filetype="GTiff", progress=1), 
                   overwrite=TRUE)

