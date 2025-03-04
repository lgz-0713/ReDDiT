import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt, student=False):
    model_opt = opt['model']
    print(model_opt['which_model_G'])
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32


    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    '''
    if opt['freq_aware']:
        model = unet.Free_UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size'],
            b1=opt['freq_awareUNet']['b1'],
            b2=opt['freq_awareUNet']['b2'],
            s1=opt['freq_awareUNet']['s1'],
            s2=opt['freq_awareUNet']['s2']
        )
    '''

    # print(model_opt['beta_schedule']['train']['n_timestep'])
    if student: 
        netG = diffusion.GaussianDiffusion(
            model,
            image_size=model_opt['diffusion']['image_size'],
            num_timesteps=model_opt['beta_schedule']['train']['n_timestep'] // 2,
            time_scale=model_opt['beta_schedule']['train']['time_scale'] * 2,
            channels=model_opt['diffusion']['channels'],
            w_gt= model_opt['diffusion']['w_gt'],
            w_snr= model_opt['diffusion']['w_snr'],
            w_str= model_opt['diffusion']['w_str'],
            w_lpips= model_opt['diffusion']['w_lpips'],
            loss_type='l1',   
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train'])
    else:
  
        netG = diffusion.GaussianDiffusion(
            model,
            image_size=model_opt['diffusion']['image_size'],
            num_timesteps=model_opt['beta_schedule']['train']['n_timestep'] ,
            time_scale=model_opt['beta_schedule']['train']['time_scale'],
            channels=model_opt['diffusion']['channels'],
            w_gt= model_opt['diffusion']['w_gt'],
            w_snr= model_opt['diffusion']['w_snr'],
            w_str= model_opt['diffusion']['w_str'],
            w_lpips= model_opt['diffusion']['w_lpips'],
            loss_type='l1',   
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train'])

    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

    
# Generator
def define_GGG(opt):
    model_opt = opt['model']
    print(model_opt['which_model_G'])
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
        model_opt['unet']['norm_groups']=32
    model = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )
    netGVar = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',  
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train']
    )
    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netGVar, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netGVar = nn.DataParallel(netGVar)
    return netGVar