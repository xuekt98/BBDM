import os
import shutil
from runners.utils import make_dir


def move_sample_files(root_dir, source_name="200", dest_name="LBBDM-f4"):
    dlist = os.listdir(os.path.join(root_dir, source_name))
    dlist.sort()
    dest_dir = os.path.join(root_dir, dest_name)
    make_dir(dest_dir)

    total = len(dlist)
    for i in range(total):
        print(f'{i}, {dlist[i]}')
        shutil.copy(os.path.join(root_dir, source_name, dlist[i], 'output_3.png'),
                    os.path.join(dest_dir, f'{dlist[i]}.png'))


move_sample_files(root_dir='/home/x/Mine/project/paper_samples/latent-diffusion-main/results/ldm-CelebAMaskHQ-f4',
                  source_name="samples",
                  dest_name="LDM-f4")
