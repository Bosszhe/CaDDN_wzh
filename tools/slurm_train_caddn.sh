#!/bin/bash


# srun --gres gpu:a100:4 --time 2-12:00:00 --job-name bevfuser_wzh --output %A.out python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher slurm --cfg_file ./cfgs/kitti_models/CaDDN.yaml --batch_size 8

srun --gres gpu:a100-80G:1 --time 2-12:00:00 --job-name caddn_kitti --output %A.out --nodelist air-node-06 python train.py --cfg_file ./cfgs/kitti_models/CaDDN_DAIR-V2X_faster_kitti_v.yaml --batch_size 4 --extra_tag faster_kitti_v
