# train:
python3 main.py --config configs/ImageNet/LBBDM-f4-inpainting.yaml --train --sample_at_start --save_top 1 --gpu_ids 0

# test:
python3 main.py --config configs/CelebAMaskHQ/LBBDM-f16.yaml --sample_to_eval --gpu_ids 1