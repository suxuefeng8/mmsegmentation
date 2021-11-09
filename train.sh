#PYTHONPATH=. python  tools/train.py celeba_mask.py

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=4 --master_port=28500 tools/train.py celeba_mask_swin_22k.py --launcher pytorch >train_swin_22k.log 2>&1 &
#CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=38500 tools/train.py celeba_mask_swin_large_22k.py --launcher pytorch >train_swin_large_22k.log 2>&1 &
