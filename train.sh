#PYTHONPATH=. python  tools/train.py celeba_mask.py

model="celeba_mask_ocr"
model="celeba_mask_swin_22k"
model="celeba_mask_large_swin_22k"
model="celeba_mask_segformer"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=4 --master_port=28500 tools/train.py ${model}.py --launcher pytorch >train_${model}.log 2>&1 &
