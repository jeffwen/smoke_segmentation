library(sf)
library(ggplot2)
library(terra)
library(raster)
library(pbapply)
library(pbmcapply)
library(fixest)
library(dplyr)

setwd("~/Stanford/projects/smoke_segmentation/")

## INPUT DATA ##
EPA_DATA_PATH <- "data/epa/"
epa_df <- list(data.frame(read.csv(paste0(EPA_DATA_PATH,"epa_ca_2019.csv"))),
    data.frame(read.csv(paste0(EPA_DATA_PATH,"epa_ca_2020.csv"))),
    data.frame(read.csv(paste0(EPA_DATA_PATH,"epa_nv_2019.csv"))),
    data.frame(read.csv(paste0(EPA_DATA_PATH,"epa_nv_2020.csv")))) %>% 
    bind_rows() %>% 
    rename_with(.fn = tolower) %>% 
    select(-c(poc, units, daily_obs_count, percent_complete)) %>% 
    rename(id=site.id, pm25=daily.mean.pm2.5.concentration, aqi=daily_aqi_value, name=site.name, 
           aqs_code=aqs_parameter_code, desc=aqs_parameter_desc, fips=state_code, 
           lat=site_latitude, lon=site_longitude) %>% 
    mutate(date=as.character(as.Date(date, "%m/%d/%Y"))) %>% 
    filter(aqs_code=="88101", source=="AQS",
           ((date>="2019-05-01" & date<="2019-10-31") | (date>="2020-05-01" & date<="2020-10-31"))) %>% 
    group_by(id, name, fips, state, county, county_code, date, source, 
             aqs_code, desc, cbsa_code, cbsa_name, lon, lat) %>% 
    summarize(pm25=mean(pm25, na.rm=T), 
              aqi=mean(aqi, na.rm=T)) %>% 
    as.data.frame() %>% 
    st_as_sf(coords=c('lon','lat'), crs=4326)

#####################
## SMOKE DATA ##
#####################

## CA NV bounding box
state_bbox <- st_as_sf(as(raster::extent(-124.48200299999999, -114.131211, 32.528832, 42.009502999999995), "SpatialPolygons"))
st_crs(state_bbox) <- 4326

## read smoke data
SMOKE_DATA_PATH <- "data/smoke_plumes/"
smoke_df <- st_read(paste0(SMOKE_DATA_PATH,"all_us_plumes_2018-2020.geojson")) %>% 
    mutate(date=as.character(as.Date(paste0(year,"-",month,"-",day)))) %>% 
    filter(year%in%c("2019","2020"), 
           month%in%c("05", "06", "07", "08", "09", "10")) %>% 
    st_crop(state_bbox)

## read in list of predicted smoke data
pred_smoke_list <- lapply(c(0,3,4,6), FUN=function(run_num){
    ## read in model smoke data
    pred_smoke_df <- st_read(paste0("~/Stanford/projects/smoke_segmentation/data/smoke_plumes/predict_test_",run_num,".geojson"))  %>% 
        mutate(date=paste0(year,"-",month,"-",day))
    
    return(pred_smoke_df)
})

#####################
## COUNT OVERLAPS ##
#####################

countSmokeOverlaps <- function(date_str, epa_data, smoke_data) {
    
    ## filter to specific day to find overlaps
    temp_epa_data <- epa_data %>% 
        filter(date==date_str) %>% 
        select(-c(date))
    temp_smoke_data <- smoke_data %>% 
        filter(date==date_str)%>% 
        select(doy)
    
    temp_epa_smoke <- st_join(temp_epa_data, temp_smoke_data, join=st_within) %>% 
        group_by(id, name, fips, state, county, county_code, source, 
                 aqs_code, desc, cbsa_code, cbsa_name, pm25, aqi) %>% 
        summarize(smoke_overlaps=sum(!is.na(doy))) %>% 
        as.data.frame() %>% 
        mutate(date=date_str)
    
    return(temp_epa_smoke)
    
}

## count smoke overlaps using smoke annotations
epa_annotation_df <- pbmclapply(X=unique(smoke_df$date), FUN=countSmokeOverlaps, epa_data=epa_df, smoke_data=smoke_df) %>% 
    bind_rows()

## count smoke overlaps using predicted plumes list
## epa doesnt have readings for every day.. so output df may be smaller
epa_smoke_pred_list <- pblapply(X=pred_smoke_list, FUN=function(pred_smoke_data){
    epa_smoke_pred_df <- pbmclapply(X=unique(pred_smoke_data$date), 
                                    FUN=countSmokeOverlaps, 
                                    epa_data=epa_df, smoke_data=pred_smoke_data) %>% 
        bind_rows()
    
    return(epa_smoke_pred_df)
})

#####################
## EVALUATE MODELS ##
#####################

## estimate fe mod for annotations
fe_mod_epa <- fixest::feols(pm25~smoke_overlaps|id, data=epa_annotation_df)  
fe_eval_epa <- fixest::r2(fe_mod_epa, type=c("r2","ar2","wr2","war2"))
fe_eval_epa <- c(fe_eval_epa, "aic"=AIC(fe_mod_epa), "bic"=BIC(fe_mod_epa), "loglik"=logLik(fe_mod_epa))

## estimate fe mod for predicted smoke
fe_eval_df <- pblapply(X=epa_smoke_pred_list, FUN=function(epa_smoke_pred_df){
    
    ## estimate fe mod
    fe_mod <- fixest::feols(pm25~smoke_overlaps|id, data=epa_smoke_pred_df)  
    
    ## evaluate performance
    fe_eval <- fixest::r2(fe_mod, type=c("r2","ar2","wr2","war2"))
    fe_eval <- c(fe_eval, "aic"=AIC(fe_mod), "bic"=BIC(fe_mod), "loglik"=logLik(fe_mod))
    
    return(fe_eval)
}) %>% 
    bind_rows() %>% 
    as.data.frame()

############################
## plot smoke annotations ##
############################
KEEP_FIPS <- c("06") # just california
states <- tigris::states()
keep_shp <- st_as_sf(states[states$STATEFP%in%KEEP_FIPS,])

## plot smoke polygon
ggplot() + 
    theme_minimal(base_size=12) + 
    theme(legend.position="right",
          plot.margin = unit(c(0, 0, 0, 0), "in"),
          panel.grid.major = element_line(colour = "gray95")) + 
    geom_sf(data=keep_shp, fill=NA) + 
    geom_sf(data=smoke_df %>% filter(year=='2020', month=='09', day=="08", start_time<"2200"), aes(fill="Smoke"), color="gray80", alpha=0.25, show.legend=T) +
    scale_fill_manual(name="Smoke", values=c("Smoke"="gray80"), labels=c("Smoke"=""),
                      guide = guide_legend(override.aes = list(linetype="blank", shape=NA))) 


ggplot() + 
    theme_minimal(base_size=12) + 
    theme(legend.position="right",
          plot.margin = unit(c(0, 0, 0, 0), "in"),
          panel.grid.major = element_line(colour = "gray95")) + 
    geom_sf(data=keep_shp, fill=NA) + 
    geom_sf(data=pred_smoke_list[[3]] %>% filter(year=='2020', month=='09', day=="08", start_time<"2205"), aes(fill="Smoke"), color="gray80", alpha=0.25, show.legend=T) +
    scale_fill_manual(name="Smoke", values=c("Smoke"="gray80"), labels=c("Smoke"=""),
                      guide = guide_legend(override.aes = list(linetype="blank", shape=NA))) 


# ## see duplicates
# epa_df[duplicated(paste0(epa_df$date,"_",epa_df$id)),]
# epa_df %>% filter(id==60510001, date=="2020-07-17")

# pbmclapply(X=c("2019-09-05"), FUN=countSmokeOverlaps, epa_data=epa_df, smoke_data=smoke_df)

# epa_mod_1_df <- pbmclapply(X=unique(smoke_df$date), FUN=countSmokeOverlaps, epa_data=epa_df, smoke_data=smoke_df) %>% 
#     bind_rows()

# match_check <- pbmclapply(unique(smoke_df$date), FUN=function(x){
#     temp_epa_blah <- epa_df %>% filter(date==x) %>% 
#         select(-c(date))
#     zing <- st_join(temp_epa_blah, 
#                     smoke_df %>% filter(date==x)%>% 
#                         select(Density, doy), 
#                     join=st_within)
#     what <- zing %>% 
#         group_by(id, name, fips, state, county, county_code, source, aqs_code, desc, cbsa_code,
#                  cbsa_name, pm25, aqi) %>% 
#         summarize(smoke_overlaps=sum(!is.na(doy))) %>% 
#         as.data.frame()
#     
#     return(data.frame(date=x, match_shape=nrow(temp_epa_blah)==nrow(what)))
#     
# }) %>% bind_rows()

