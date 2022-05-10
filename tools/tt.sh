#!/bin/bash


srun --gres gpu:a100:4 --time 2-12:00:00 --job-name bevfuser_wzh --output %A_%a.out python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher slurm ----cfg_file ./cfgs/kitti_models/CaDDN.yaml
