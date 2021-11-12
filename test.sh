#PYTHONPATH=. python  tools/train.py celeba_mask.py

set -eux

model="celeba_mask_ocr_1024"
#model="celeba_mask_swin_22k"
#model="celeba_mask_swin_large_22k"
#model="celeba_mask_segformer"

test_epoch="1"
checkpoint="work_dirs/"${model}"/epoch_"${test_epoch}".pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=4 --master_port=36500 tools/test.py "configs/celeba_mask/"${model}.py $checkpoint --eval mIoU --launcher pytorch >test_${model}.log 2>&1 &
