#PYTHONPATH=. python  tools/train.py celeba_mask.py

set -eux

#model="celeba_mask_ocr"
#model="celeba_mask_swin_22k"
model="celeba_mask_swin_large_22k"
#model="celeba_mask_segformer"

test_epoch="33"
checkpoint="work_dirs/"${model}"/epoch_"${test_epoch}".pth"

visualize_dir="visualize/"

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=36500 tools/test.py ${model}.py $checkpoint --show-dir $visualize_dir --launcher pytorch >visualize_${model}.log 2>&1 &
