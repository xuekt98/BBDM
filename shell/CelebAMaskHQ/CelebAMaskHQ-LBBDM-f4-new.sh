#train
### python3 main.py --config configs/Template_LBBDM_f4.yaml --train --sample_at_start --save_top 1 --gpu_ids 0 \
### --resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

#test
python3 main.py --config configs/CelebAMaskHQ/LBBDM-CelebAMaskHQ-f4-new.yaml --gpu_ids 0 \
--resume_model results/CelebAMaskHQ/LBBDM-f4-new/checkpoint/latest_model_110.pth \
# --sample_to_eval

#preprocess and evaluation
## rename
#python3 preprocess_and_evaluation.py -f rename_samples \
#-r results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval \
#-s 200 \
#-t 200_rename

## copy
#python3 preprocess_and_evaluation.py -f copy_samples \
#-r results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval \
#-s 200_rename \
#-t LBBDM-f4-new

## LPIPS
#python3 preprocess_and_evaluation.py -f LPIPS -n 5 \
#-s results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval/200_rename \
#-t results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval/ground_truth

## max_min_LPIPS
### python3 preprocess_and_evaluation.py -f max_min_LPIPS -s source/dir -t target/dir -n 1

## diversity
#python3 preprocess_and_evaluation.py -f diversity -n 5 \
#-s results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval/200_rename

## fidelity
#fidelity --gpu 0 --fid --input1 results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval/LBBDM-f4-new \
#--input2 results/CelebAMaskHQ/LBBDM-f4-new/sample_to_eval/ground_truth
