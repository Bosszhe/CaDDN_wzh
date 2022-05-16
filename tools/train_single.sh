#!/bin/bash


# srun --gres gpu:a100:4 --time 2-12:00:00 --job-name bevfuser_wzh --output %A.out python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher slurm --cfg_file cfgs/kitti_models/CaDDN.yaml --batch_size 8

# srun --gres gpu:a100-80G:1 --time 4-00:00:00 --job-name caddn_kitti --output %A.out --nodelist air-node-06 python train.py --cfg_file cfgs/kitti_models/CaDDN_DAIR-V2X_faster_kitti_v.yaml --batch_size 2 --extra_tag faster_kitti_v

# srun --gres gpu:a100-80G:1 --time 6-00:00:00 --job-name caddn_kitti_v --output %A.out python train.py --cfg_file cfgs/kitti_models/CaDDN_DAIR-V2X_kitti_v.yaml --batch_size 2 

# srun --gres gpu:a100-80G:1 --time 1-00:00:00 --job-name caddn_v_test --output %A.out python test.py --cfg_file cfgs/kitti_models/CaDDN_DAIR-V2X_kitti_v.yaml --batch_size 1 --ckpt ../checkpoints/checkpoint_epoch_35.pth


srun --gres gpu:a100-80G:1 --time 1-00:00:00 --job-name caddn_kitti_test --output %A.out python test.py --cfg_file cfgs/kitti_models/CaDDN_DAIR-V2X_kitti_v.yaml --batch_size 1 --ckpt ../checkpoints/checkpoint_epoch_35.pth