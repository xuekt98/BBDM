import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        # total_num, trainable_num = get_parameter_number(net.vqgan)
        # print("Total Number of VQGAN parameter: %.2fM" % (total_num / 1e6))
        # print("Trainable Number of VQGAN parameter: %.2fM" % (trainable_num / 1e6))
        # total_num, trainable_num = get_parameter_number(net.denoise_fn)
        # print("Total Number of BrownianBridge parameter: %.2fM" % (total_num / 1e6))
        # print("Trainable Number of BrownianBridge parameter: %.2fM" % (trainable_num / 1e6))
        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name) = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))
        inversion_sample_path = make_dir(os.path.join(sample_path, f'inversion_sample'))
        inversion_one_step_path = make_dir(os.path.join(sample_path, 'inversion_one_step_samples'))

        (x, x_name), (x_cond, x_cond_name) = batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        # samples, one_step_samples = net.sample(x_cond,
        #                                        clip_denoised=self.config.testing.clip_denoised,
        #                                        sample_mid_step=True)
        # self.save_images(samples, reverse_sample_path, grid_size, save_interval=20,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        #
        # self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=20, head_threshold=190,
        #                  tail_threshold=10, writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
        #
        # sample = samples[-1]
        sample = net.sample(x_cond, clip_denoised=self.config.testing.clip_denoised).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

        # inversion_samples, inversion_one_step_samples = net.inversion_sample(x,
        #                                                                      clip_denoised=self.config.testing.clip_denoised,
        #                                                                      sample_mid_step=True)
        # self.save_images(inversion_samples, inversion_sample_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_inversion_sample' if stage != 'test' else None)
        #
        # self.save_images(inversion_one_step_samples, inversion_one_step_path, grid_size, save_interval=200, head_threshold=990,
        #                  tail_threshold=10, writer_tag=f'{stage}_inversion_one_step_sample' if stage != 'test' else None)

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        condition_path = make_dir(os.path.join(sample_path, f'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        for test_batch in pbar:
            (x, x_name), (x_cond, x_cond_name) = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            for j in range(sample_num):
                # sample = net.sample(x_cond, clip_denoised=False)
                sample = net.sample_vqgan(x)
                for i in range(batch_size):
                    condition = x_cond[i].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    if j == 0:
                        save_single_image(condition, condition_path, f'{x_cond_name[i]}.png', to_normal=to_normal)
                        save_single_image(gt, gt_path, f'{x_name[i]}.png', to_normal=to_normal)
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.png', to_normal=to_normal)
