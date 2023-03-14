import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():
            x_latent = self.encode(x)
            x_cond_latent = self.encode(x_cond)
        context = self.get_cond_stage_context(x_cond)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x):
        model = self.vqgan
        x_latent = model.encoder(x)
        # if not self.model_config.latent_before_quant_conv:
        x_latent = model.quant_conv(x_latent)
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent):
        model = self.vqgan
        # if self.model_config.latent_before_quant_conv:
        # x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach())
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach())
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent)
            return out

    @torch.no_grad()
    def sample_perturbed_image(self, x, x_cond, input_step=None, with_cond=True):
        x_latent = self.encode(x)
        x_cond_latent = self.encode(x_cond)

        step = random.randint(1, self.num_timesteps) if input_step is None else input_step
        t = torch.full((x_latent.shape[0],), step).to(x_latent.device)
        x_t, objective = self.brownianbridge.q_sample(x_latent, x_cond_latent, t)

        if not with_cond:
            x_cond_zero = torch.zeros_like(x_cond).to(x_cond.device)
            objective_recon = self.brownianbridge.denoise_fn(x_t, timesteps=t,
                                                             context=self.brownianbridge.cond_stage_model(x_cond_zero))
        else:
            objective_recon = self.brownianbridge.denoise_fn(x_t, timesteps=t,
                                                             context=self.brownianbridge.cond_stage_model(x_cond))

        x0_latent_recon = self.brownianbridge.predict_x0_from_noise(x_t, x_cond_latent, t, objective_recon)
        x0_diffusion_recon = self.decode(x0_latent_recon.detach(), cond=False)

        x0_vqgan_recon = self.decode(x_latent.detach(), cond=False)
        x_t_recon = self.decode(x_t.detach(), cond=False)
        return x0_diffusion_recon, x0_vqgan_recon, x_t_recon, step

    @torch.no_grad()
    def sample_mid_steps(self, x_cond, clip_denoised=False):
        x_cond_latent = self.encode(x_cond)
        temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                 context=self.get_cond_stage_context(x_cond),
                                                 clip_denoised=clip_denoised,
                                                 sample_mid_step=True)
        out_samples = []
        for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                      smoothing=0.01):
            x_latent = temp[i]
            with torch.no_grad():
                out = self.decode(x_latent.detach(), cond=False)
            out_samples.append(out.to('cpu'))

        one_step_samples = []
        for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps", dynamic_ncols=True,
                      smoothing=0.01):
            x_latent = one_step_temp[i]
            with torch.no_grad():
                out = self.decode(x_latent.detach(), cond=False)
            one_step_samples.append(out.to('cpu'))
        return out_samples, one_step_samples

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    @torch.no_grad()
    def reverse_sample(self, x, skip=False):
        x_ori_latent = self.vqgan.encoder(x)
        temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
        x_latent = temp[-1]
        x_latent = self.vqgan.quant_conv(x_latent)
        x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
        out = self.vqgan.decode(x_latent_quant)
        return out
