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
runs <- c(0,3,4,6)
train_df <- lapply(runs, FUN=dataFormatter, data="train") %>% 
    bind_rows() %>% 
    mutate(run=ordered(run, levels=runs))
val_df <- lapply(runs, FUN=dataFormatter, data="val") %>% 
    bind_rows() %>% 
    mutate(run=ordered(run, levels=runs),
           id=((id - min(val_df$id)) * 
                   (max(train_df$id) - min(train_df$id)))/
               (max(id) - min(id)) + min(train_df$id)) 

plot_df <- bind_rows(train_df, val_df)

## plot accuracy curves
num_breaks <- 5
p1 <- ggplot(data=plot_df, aes(linetype=Data)) + 
    theme_minimal(base_size=11) + 
    theme(legend.position="bottom") + 
    geom_line(aes(x=id, y=avg_acc, color=run), data=train_df) + 
    geom_line(aes(x=id, y=avg_acc, color=run), data=val_df) + 
    lims(y=c(0,0.15)) + 
    scale_x_continuous(breaks=quantile(train_df$id, probs=c(1:num_breaks)/num_breaks), 
                       labels=seq(1,21,num_breaks)) + 
    scale_color_discrete(name="Run", labels=c("1 band", "3 band", "4 band", "3 band loss sampling")) +
    labs(y='', x='Epoch', color='Run', subtitle='Avg. Dice coefficient') + 
    guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2))


## plot loss curves
p2 <- ggplot(data=plot_df, aes(linetype=Data)) + 
    theme_minimal(base_size=11) +
    theme(legend.position="none") + 
    geom_line(aes(x=id, y=avg_loss, color=run), data=train_df) + 
    geom_line(aes(x=id, y=avg_loss, color=run), data=val_df) + 
    scale_x_continuous(breaks=quantile(train_df$id, probs=c(1:num_breaks)/num_breaks), 
                       labels=seq(1,21,num_breaks)) + 
    scale_color_discrete(name="Run", labels=c("1 band", "3 band", "4 band", "3 band loss sampling")) + 
    lims(y=c(0.1, 0.7))+
    labs(y='', x='Epoch', color='Run', subtitle='Avg. Binary cross entropy loss') + 
    guides(color=guide_legend(nrow=2), linetype=guide_legend(nrow=2))


p_final <- cowplot::plot_grid(p1, p2, ncol=1)
cowplot::save_plot("etc/figures/training_curves.png", p_final, base_width=5, base_height=7)

