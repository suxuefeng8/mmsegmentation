#PYTHONPATH=. python  tools/train.py celeba_mask.py

set -eux

#model="celeba_mask_ocr"
#model="celeba_mask_swin_22k"
model="celeba_mask_swin_large_22k"
#model="celeba_mask_segformer"

test_epoch="33"
checkpoint="work_dirs/"${model}"/epoch_"${test_epoch}".pth"

visualize_dir="visualize/"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py ${model}.py "configs/celeba_mask/"$checkpoint --show-dir $visualize_dir >visualize_${model}.log 2>&1 &
