python train_Unet.py ^
--model_path "..\results\cmc\s2\memory_nce_4096_resnet18_lr_0.03_decay_0.0001_bsz_32_aug_True\epoch=399-train_loss=2.55.ckpt" ^
--save_path ..\results\unet\ ^
--dataset_name s2 ^
--image_folder ..\data\tiles\supervised\s2\images ^
--label_folder ..\data\tiles\supervised\s2\mapped_labels ^
--train_image_list ..\data\tiles\supervised\s2\filenames.txt ^
--train_batch_size 16 ^
--test_batch_size 8 ^
--n_classes 12 ^
--channels_l 1 7 8 9 10 ^
--channels_ab 2 3 4 5 6 ^
--augment ^
--gpu 1