library(ggplot2)
library(cowplot)

setwd("~/Stanford/projects/smoke_segmentation/")

## read in training log data
dataFormatter <- function(run, data, avg_window=500){
    
    ## read in data for specific train or val run
    df <- read.table(pipe(paste0("awk '/epoch:/ && /",data,
                                 "/ {print}' < ~/Stanford/projects/smoke_segmentation/training_logs/run_",
                                 run,"_training_log.txt")), sep="|")
                     
    df <- df %>% 
        tidyr::separate(V1, into=c(NA, "epoch"), sep=": ") %>% 
        tidyr::separate(V2, into=c(NA, "batch"), sep=": ") %>% 
        tidyr::separate(V3, into=c(NA, "loss"), sep=": ") %>% 
        tidyr::separate(V4, into=c(NA, "acc"), sep=": ") %>% 
        tidyr::separate(V5, into=c(NA, "iou"), sep=": ") %>% 
        mutate(across(.fns = as.numeric)) %>% 
        mutate(Data=data, run=run, id = row_number())
    
    ## calculate rolling avg
    df$avg_loss <- zoo::rollapply(df$loss, width = avg_window, FUN = mean, align = "right", partial=T)
    df$avg_acc <- zoo::rollapply(df$acc, width = avg_window, FUN = mean, align = "right", partial=T)
    df$avg_iou <- zoo::rollapply(df$iou, width = avg_window, FUN = mean, align = "right", partial=T)
                     
    return(df)      
}

## combine training run logs for training and val 
runs <- c(4,5)
train_df <- lapply(runs, FUN=dataFormatter, data="train") %>% 
    bind_rows() %>% 
    mutate(run=ordered(run, levels=runs))
val_df <- lapply(runs, FUN=dataFormatter, data="val") %>% 
    bind_rows() %>% 
    mutate(run=ordered(run, levels=runs))

## plot training curves
num_breaks <- 5
ggplot(data=train_df) + 
    theme_minimal(base_size=12) + 
    geom_line(aes(x=id, y=avg_acc, color=run), size=1) + 
    scale_x_continuous(breaks=quantile(train_df$id, probs=c(1:num_breaks)/num_breaks), 
                       labels=seq(1,21,num_breaks)) + 
    labs(y='Avg. dice coefficient', x='Epoch', color='Run')

