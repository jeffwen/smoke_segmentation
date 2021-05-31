#!/bin/sh
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_0.geojson --batch_size=1 --model=../checkpoints/best_run0.pth.tar --bands true_color
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_1.geojson --batch_size=1 --model=../checkpoints/best_run1.pth.tar --bands true_color
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_3.geojson --batch_size=1 --model=../checkpoints/best_run3.pth.tar --bands true_color C07 C11
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_5.geojson --batch_size=1 --model=../checkpoints/best_run5.pth.tar --bands true_color C07 C11
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_4.geojson --batch_size=1 --model=../checkpoints/best_run4.pth.tar --bands true_color C07 C11 merra2
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_8.geojson --batch_size=1 --model=../checkpoints/best_run8.pth.tar --bands true_color C07 C11 merra2
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_6.geojson --batch_size=1 --model=../checkpoints/best_run6.pth.tar --bands true_color C07 C11
python test.py ../data/img --out_file=../data/smoke_plumes/predict_test_7.geojson --batch_size=1 --model=../checkpoints/best_run7.pth.tar --bands true_color C07 C11

#python train.py ../data/crops_unfiltered/file_index.csv --batch-size=16 --epochs=22 --run=1 --lr=0.00005 --bands true_color
#python train.py ../data/crops_unfiltered/file_index.csv --batch-size=16 --epochs=22 --run=7 --lr=0.00005 --bands true_color C07 C11 --loss-sampling --loss-sample-k=1
#python train.py ../data/crops_unfiltered/file_index.csv --batch-size=16 --epochs=22 --run=8 --lr=0.00005 --bands true_color C07 C11 merra2
