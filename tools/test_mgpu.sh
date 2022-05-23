

###################### dist_test scripts
#!/usr/bin/env bash


python -m torch.distributed.launch --nproc_per_node 2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/CaDDN_DAIR-V2X_kitti_v.yaml --extra_tag epoch_80_gpu_2 --batch_size 2 --eval_all
