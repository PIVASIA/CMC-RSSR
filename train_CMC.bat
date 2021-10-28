python train_CMC.py --image_folder ..\data\tiles\s2^
 --image_list ..\data\tiles\s2\filenames.txt^
 --model resnet18^
 --model_path ..\results\cmc\s2^
 --batch_size 32^
 --dataset_name s2^
 --channels_l 1 7 8 9 10^
 --channels_ab 2 3 4 5 6^
 --image_aug^
 --gpu 1