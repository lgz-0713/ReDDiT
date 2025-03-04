import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import cv2
import torchvision.transforms as T

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import lpips
from torchvision.utils import save_image
from torch.optim.swa_utils import AveragedModel

 

transform = T.Lambda(lambda t: (t + 1) / 2)

def extract(v, t, x_shape):
   
    try:
        out = torch.gather(v, index=t, dim=0).float()
    except:
        # import pdb; pdb.set_trace()
        print(t)
        # import pdb; pdb.set_trace()
        print(print(v.shape))
        # import pdb; pdb.set_trace()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))



def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        num_timesteps,
        time_scale,
        w_str,
        w_gt,
        w_snr,
        w_lpips,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.num_timesteps = num_timesteps
        device = torch.device("cuda")

        self.w_str = w_str 
        self.w_gt = w_gt
        self.w_snr = w_snr
        self.w_lpips = w_lpips
        # self.lpips = lpips.LPIPS(net='vgg').cuda()
        # print(self.num_timesteps)
        # import pdb; pdb.set_trace()
        self.time_scale = time_scale
        self.CD = False
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt, device)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep= self.num_timesteps* self.time_scale + 1,
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))
        
        timesteps, = betas.shape
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
    
    def predict_eps_from_x(self, x_t, x_0, t):

        eps = (x_t -self.sqrt_alphas_cumprod[t] * x_0) / self.sqrt_one_minus_alphas_cumprod[t]
        return eps
    
    def predict_eps(self, x_t, x_0, continuous_sqrt_alpha_cumprod):
        
        eps = (1. / (1 - continuous_sqrt_alpha_cumprod **2).sqrt()) * x_t - \
            (1. / (1 - continuous_sqrt_alpha_cumprod**2) -1).sqrt() * x_0

        return eps
    
    def predict_start(self, x_t, continuous_sqrt_alpha_cumprod, noise):

        return (1. / continuous_sqrt_alpha_cumprod) * x_t - \
            (1. / continuous_sqrt_alpha_cumprod**2 - 1).sqrt() * noise

    def predict_t_minus1(self, x, t, continuous_sqrt_alpha_cumprod, noise, clip_denoised=True):

        x_recon = self.predict_start(x, 
                    continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), 
                    noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        noise_z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + noise_z * (0.5 * model_log_variance).exp() 

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+self.time_scale]]).repeat(batch_size, 1).to(x.device)
        
        eps = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)[0]
        # print(t)

        x_recon = self.predict_start_from_noise(x, t=t*self.time_scale, noise=eps)


        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, eps

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, eps = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            # print(self.time_scale)
            # import pdb; pdb.set_trace()
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, stride=1):
        return self.ddim(x_in, continous, stride=stride)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):

        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def ddim(self, x_in, continous=False, snr_aware=False, stride=1, clip_denoised=True): 
        x = x_in
        condition_x = x_in 
        x_t = torch.randn(x.shape, device=x.device)

        batch_size = x_in.shape[0]
  
        for time_step in reversed(range(stride, self.num_timesteps + 1, stride)):

            
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            s = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * (time_step - stride)
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t* self.time_scale]]).repeat(batch_size, 1).to(x.device)
            eps =self.denoise_fn(torch.cat([condition_x, x_t], dim=1), noise_level)[0]
            x_0 = self.predict_start_from_noise(x_t, t * self.time_scale, eps)
            if clip_denoised:
                x_0 = torch.clip(x_0, -1., 1.)
                eps = self.predict_eps_from_x(x_t, x_0, t * self.time_scale)

            x_t = self.sqrt_alphas_cumprod[s * self.time_scale] * x_0 + self.sqrt_one_minus_alphas_cumprod[s * self.time_scale] * eps

        return torch.clip(x_t, -1, 1)

    def SNR_map(self, x_0):
        blur_transform = T.GaussianBlur(kernel_size=15, sigma=3)
        blur_x_0 = blur_transform(x_0)
        gray_blur_x_0 = blur_x_0[:, 0:1, :, :] * 0.299 + blur_x_0[:, 1:2, :, :] * 0.587 + blur_x_0[:, 2:3, :, :] * 0.114
        gray_x_0 = x_0[:, 0:1, :, :] * 0.299 + x_0[:, 1:2, :, :] * 0.587 + x_0[:, 2:3, :, :] * 0.114
        noise = torch.abs(gray_blur_x_0 - gray_x_0)

        return noise
    

        
    def loss(self, x_in, student, noise=None, lpips_func=None):
        x_0 = x_in['GT']
        [b, c, h, w] = x_0.shape
      
        t = 2 * np.random.randint(1, student.num_timesteps + 1) 

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[(t-1)*self.time_scale],
                self.sqrt_alphas_cumprod_prev[t*self.time_scale],
                size=b
            )
        ).to(x_0.device)

        continuous_sqrt_alpha_cumprod_t_mins_1 = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[(t-2)*self.time_scale],
                self.sqrt_alphas_cumprod_prev[(t-1)*self.time_scale],
                size=b
            )
        ).to(x_0.device)
        continuous_sqrt_alpha_cumprod_t_mins_2 = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[(t-3)*self.time_scale],
                self.sqrt_alphas_cumprod_prev[(t-2)*self.time_scale],
                size=b
            )
        ).to(x_0.device)
       
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        continuous_sqrt_alpha_cumprod_t_mins_1 = continuous_sqrt_alpha_cumprod_t_mins_1.view(b, -1)
        continuous_sqrt_alpha_cumprod_t_mins_2 = continuous_sqrt_alpha_cumprod_t_mins_2.view(b, -1)
        
        noise = default(noise, lambda: torch.randn_like(x_0))
        t = torch.tensor([t], dtype=torch.int64).to(x_0.device)
        bs = x_0.size(0)

        with torch.no_grad():   
            z_t = self.q_sample(x_0, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise)
            eps_rec, _ = self.denoise_fn(torch.cat([x_in['LQ'], z_t], dim=1), continuous_sqrt_alpha_cumprod) 
            x_0_rec = self.predict_start(z_t, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), eps_rec)
            z_t_minus_1 = self.q_sample(x_0_rec, continuous_sqrt_alpha_cumprod_t_mins_1.view(-1, 1, 1, 1), eps_rec)
            eps_rec_rec, _ = self.denoise_fn(torch.cat([x_in['LQ'], z_t_minus_1], dim=1), continuous_sqrt_alpha_cumprod_t_mins_1)
            x_0_rec_rec = self.predict_start(z_t_minus_1, continuous_sqrt_alpha_cumprod_t_mins_1.view(-1, 1, 1, 1), eps_rec_rec)
            z_t_minus_2 = self.q_sample(x_0_rec_rec, continuous_sqrt_alpha_cumprod_t_mins_2.view(-1, 1, 1, 1), eps_rec_rec)
            frac = (1 - continuous_sqrt_alpha_cumprod_t_mins_2.view(-1, 1, 1, 1)**2).sqrt() / (1- continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)**2).sqrt()         
            if self.w_snr != 0:
                y = x_in['LQ']
                T,_=torch.max(y,dim=1, keepdim=True)
                T=T+0.1
                y = y / T 
                iso_noise = self.SNR_map(y)
                y = y - iso_noise      
                refine_x_0 =  y                  
                z_t_minus_2_refine = self.q_sample(refine_x_0, continuous_sqrt_alpha_cumprod_t_mins_2.view(-1, 1, 1, 1), eps_rec_rec)
                z_t_minus_2 =  z_t_minus_2 + self.w_snr *(z_t_minus_2_refine - z_t_minus_2)
            
            x_target = (z_t_minus_2 - frac * z_t) / ( continuous_sqrt_alpha_cumprod_t_mins_2.view(-1, 1, 1, 1) - frac * continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1 ))
            eps_target = self.predict_eps(z_t, x_target, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)) 

        eps_predicted, _ = student.denoise_fn(torch.cat([x_in['LQ'], z_t], dim=1), continuous_sqrt_alpha_cumprod)
        x_0_predicted = self.predict_start(z_t, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), eps_predicted)
        loss_x_0 = torch.mean(F.mse_loss(x_0_predicted, x_target, reduction='none').reshape(bs, -1), dim=-1)
        loss_eps = torch.mean(F.mse_loss(eps_predicted, eps_target, reduction='none').reshape(bs, -1), dim=-1)

        loss_stru = torch.zeros_like(loss_x_0) # 0.
        if self.w_gt != 0:
            loss_output_x0 = torch.mean(F.mse_loss(x_0, x_0_predicted, reduction='none').reshape(bs, -1), dim=-1)  
            loss_output_eps = torch.mean(F.mse_loss(noise, eps_predicted, reduction='none').reshape(bs, -1), dim=-1)  

        else:
            loss_output_x0 = torch.zeros_like(loss_x_0) # 0.
            loss_output_eps = torch.zeros_like(loss_eps) # 0.

        if self.w_lpips != 0:
            loss_lpips = torch.mean(lpips_func(x_0, x_0_predicted)) 
        else:
            loss_lpips = torch.zeros_like(loss_x_0) # 0.

        return torch.mean(torch.maximum(loss_x_0, loss_eps)) + \
              self.w_gt * torch.mean(torch.maximum(loss_output_x0, loss_output_eps)) + \
              self.w_lpips*torch.mean(loss_lpips) + \
              self.w_str*torch.mean(loss_stru)

    
    


    def forward(self, x, s_model=None,  *args, **kwargs):
            return self.loss(x, s_model, *args, **kwargs)


