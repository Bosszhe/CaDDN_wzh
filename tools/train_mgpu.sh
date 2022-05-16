#!/bin/bash


# SBATCH --job-name bevfuser_wzh                  # 任务名叫 example
# #SBATCH --array 0-99                        # 提交 100 个子任务，序号分别为 0,1,2,...99
# SBATCH --gres gpu:a100:8                   # 每个子任务都用一张 A100 GPU
# SBATCH --time 1-12:00:00                    # 子任务 1 天 1 小时就能跑完
# SBATCH --output %A_%a.out                  # 100个程序输出重定向到 [任务id]_[子任务序号].out
# SBATCH --mail-user wangzhe@air.tsinghua.edu.cn       # 这些程序开始、结束、异常突出的时候都发邮件告诉我
# SBATCH --mail-type ALL                     

# # 任务 ID 通过 SLURM_ARRAY_TASK_ID 环境变量访问
# # 上述行指定参数将传递给 sbatch 作为命令行参数
# # 中间不可以有非 #SBATCH 开头的行

# # 执行 sbatch 命令前先通过 conda activate [env_name] 进入环境



#####################

# srun --gres gpu:a100:4 --time 2-12:00:00 --job-name bevfuser_wzh python -m torch.distributed.launch --nproc_per_node=8 train.py --launcher slurm ----cfg_file ./cfgs/kitti_models/CaDDN.yaml

#####################
# set -x
# NGPUS=$1
# PY_ARGS=${@:2}

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --batch_size 2 --launcher pytorch ${PY_ARGS} 

#####################

# CUDA_VISIBLE_DEVICES=3,6 python -m torch.distributed.launch --nproc_per_node=2 train.py --batch_size 4 --launcher pytorch --cfg_file cfgs/kitti_models/CaDDN.yaml --extra_tag DEBUG

#####################

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/kitti_models/CaDDN_DAIR-V2X_kitti_v.yaml --batch_size 2 --extra_tag DEBUG