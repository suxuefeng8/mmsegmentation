#PYTHONPATH=. python  tools/train.py celeba_mask.py

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=28500 tools/train.py celeba_mask_segformer.py --launcher pytorch >train_segformer.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 tools/train.py celeba_mask_ocr.py --launcher pytorch >train_ocr.log 2>&1 &
