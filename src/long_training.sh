#!/bin/sh
python train.py ../data/crops_unfiltered/file_index.csv --batch-size=16 --epochs=22 --run=1 --lr=0.00005 --bands true_color
python train.py ../data/crops_unfiltered/file_index.csv --batch-size=16 --epochs=22 --run=7 --lr=0.00005 --bands true_color C07 C11 --loss-sampling --loss-sample-k=1
python train.py ../data/crops_unfiltered/file_index.csv --batch-size=16 --epochs=22 --run=8 --lr=0.00005 --bands true_color C07 C11 merra2
