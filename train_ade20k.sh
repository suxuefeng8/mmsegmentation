#PYTHONPATH=. python  tools/train.py celeba_mask.py

config="configs/swin/ade_swin_large_22k.py"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=4 --master_port=28500 tools/train.py $config --launcher pytorch >train_ade.log 2>&1 &
