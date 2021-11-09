#PYTHONPATH=. python  tools/train.py celeba_mask.py

set -eux

#model="celeba_mask_ocr"
#model="celeba_mask_swin_22k"
#model="celeba_mask_large_swin_22k"
model="celeba_mask_segformer"

test_epoch="31"
checkpoint="work_dirs/"${model}"/epoch_"${test_epoch}".pth"

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=38500 tools/test.py ${model}.py $checkpoint --eval mIoU --launcher pytorch >test_${model}.log 2>&1 &
