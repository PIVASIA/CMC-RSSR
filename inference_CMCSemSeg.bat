python inference_CMCSemSeg.py^
 --image_folder ..\data\tiles\supervised\s2\images^
 --test_image_list ..\data\tiles\supervised\s2\filenames.txt^
 --label_folder ..\data\tiles\supervised\s2\mapped_labels^
 --model resnet18^
 --model_path ..\results\cmc\s2\memory_nce_4096_resnet18_lr_0.03_decay_0.0001_bsz_64_aug_True\epoch=398-train_loss=0.03-val_loss=0.03.ckpt^
 --train_batch_size 64^
 --test_batch_size 32^
 --dataset_name s2^
 --channels_l 1 7 8 9 10^
 --channels_ab 2 3 4 5 6^
 --augment^
 --gpu 1