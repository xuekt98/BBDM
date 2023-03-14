import os
from abc import ABC
from PIL import Image
from tqdm.autonotebook import tqdm
from runners.BaseRunner import BaseRunner
from runners.utils import get_image_grid


class DiffusionBaseRunner(BaseRunner, ABC):
    def __init__(self, config):
        super().__init__(config)

    def save_images(self, all_samples, sample_path, grid_size=4, gif_interval=-1, save_interval=100,
                    head_threshold=10000, tail_threshold=0, writer_tag=None):
        """
        save diffusion mid-step images
        :param all_samples: all samples
        :param sample_path: sample path
        :param grid_size: grid size
        :param gif_interval: gif interval; if gif_interval >= 0, save gif frame every gif_interval
        :param save_interval: interval of saving image
        :param head_threshold: save all samples in range [T, head_threshold]
        :param tail_threshold: save all samples in range [0, tail_threshold]
        :param writer_tag: if writer_tag is not None, write output image to tensorboard with tag=writer_tag
        :return:
        """
        dataset_config = self.config.data.dataset_config
        batch_size = all_samples[-1].shape[0]
        imgs = []
        for i, sample in enumerate(tqdm(all_samples, total=len(all_samples), desc='saving images')):
            if (gif_interval > 0 and i % gif_interval == 0) or i % save_interval == 0 or i > head_threshold or i < tail_threshold:
                sample = sample.view(batch_size, dataset_config.channels,
                                     dataset_config.image_size, dataset_config.image_size)

                image_grid = get_image_grid(sample, grid_size, to_normal=dataset_config.to_normal)
                # if self.config.task == 'colorization':
                #     image_grid = cv2.cvtColor(image_grid, cv2.COLOR_LAB2RGB)
                im = Image.fromarray(image_grid)
                if gif_interval > 0 and i % gif_interval == 0:
                    imgs.append(im)

                if i % save_interval == 0 or i > head_threshold or i < tail_threshold:
                    im.save(os.path.join(sample_path, 'image_{}.png'.format(i)))

        image_grid = get_image_grid(all_samples[-1], grid_size, to_normal=dataset_config.to_normal)
        # if self.config.task == 'colorization':
        #     image_grid = cv2.cvtColor(image_grid, cv2.COLOR_LAB2RGB)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'image_out.png'))

        if writer_tag is not None:
            self.writer.add_image(writer_tag, image_grid, self.global_step, dataformats='HWC')

        if gif_interval > 0:
            imgs[0].save(os.path.join(sample_path, "movie.gif"), save_all=True, append_images=imgs[1:],
                         duration=1, loop=0)
