import os
import shutil
import re
from runners.utils import make_dir


def move_BBDM_sample_files(data_dir):
    dir_list = os.listdir(os.path.join(data_dir, 'without_diff'))
    dir_list.sort()
    dest = os.path.join(data_dir, 'BBDM2-f4')
    make_dir(dest)

    total = len(dir_list)
    for i in range(total):
        print(f'{i}, {dir_list[i]}')
        shutil.copy(os.path.join(data_dir, 'without_diff', dir_list[i], 'output_1.png'), os.path.join(dest, f'{dir_list[i]}.png'))


def move_ldm_files(data_dir):
    root_dir = '/home/x/Mine/project/paper_samples/latent-diffusion-main/results'
    dir_list = os.listdir(os.path.join(root_dir, data_dir, 'samples'))
    dir_list.sort()
    dest = os.path.join(root_dir, data_dir, data_dir)
    make_dir(dest)

    total = len(dir_list)
    for i in range(total):
        print(f'{i}, {dir_list[i]}')
        shutil.copy(os.path.join(root_dir, data_dir, 'samples', dir_list[i], 'output_0.png'), os.path.join(dest, f'{dir_list[i]}.png'))


move_ldm_files('ldm-CelebAMaskHQ-f4')
