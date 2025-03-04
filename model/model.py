import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ema import EMA
from torch.optim import lr_scheduler
import lpips

logger = logging.getLogger('base')
skip_para = []

skip_para = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 
             'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod',
               'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 
               'posterior_mean_coef1', 'posterior_mean_coef2',]

def get_scheduler(optimizer, opt):
    if opt['train']["optimizer"]['lr_policy'] == 'linear':
        def lambda_rule(iteration):
            lr_l = 1.0 - max(0, iteration-opt['train']["optimizer"]["n_lr_iters"]) / float(opt['train']["optimizer"]["lr_decay_iters"] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt['train']["optimizer"]['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt['train']["optimizer"]["lr_decay_iters"], gamma=0.8)
    elif opt['train']["optimizer"]['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt['train']["optimizer"]['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt['train']["optimizer"]['lr_policy'])
    return scheduler


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        if opt['dist']:
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt, student=False))
        if opt['dist']:
            self.netG.to(device)
       
        # self.netG.to(device)

        self.schedule_phase = None
        self.opt = opt

        # set loss and load resume state
        self.set_loss()

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()

        if self.opt['phase'] == 'test':
            self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)

        else:
            self.load_network()
            if opt['dist']:
                self.netG = DDP(self.netG, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)


    def feed_data(self, data):

        dic = {}

        if self.opt['dist']:
            dic = {}
            dic['LQ'] = data['LQ'].to(self.local_rank)
            dic['GT'] = data['GT'].to(self.local_rank)
            self.data = dic
        else:
            dic['LQ'] = data['LQ']
            dic['GT'] = data['GT']

            self.data = self.set_device(dic)


    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['LQ'], continous)
                
            else:
                if self.opt['dist']:
                    self.SR = self.netG.module.super_resolution(self.data['LQ'], continous)
                else:
                    self.SR = self.netG.super_resolution(self.data['LQ'], continous)

        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):

        if self.opt['dist']:

            device = torch.device("cuda", self.local_rank)
            if self.schedule_phase is None or self.schedule_phase != schedule_phase:
                self.schedule_phase = schedule_phase
                if isinstance(self.netG, nn.DataParallel):
                    self.netG.module.set_new_noise_schedule(
                        schedule_opt, self.device)
                else:
                    self.netG.set_new_noise_schedule(schedule_opt)

        else:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                # self.netG.set_new_noise_schedule(schedule_opt, self.device)
                self.netG.set_new_noise_schedule(schedule_opt, self.device)


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['HQ'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['LQ'].detach().float().cpu()
            out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
            else:
                out_dict['LQ'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(s)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))

    def save_network(self, distill_step, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'num_step_{}', 'I{}_E{}_gen.pth'.format(distill_step, iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'num_step_{}', 'I{}_E{}_opt.pth'.format(distill_step, iter_step, epoch))
        
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)



        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}'.format(load_path)
            
            # gen
            networks = [self.netG, self.netG]
            for network in networks:
                if isinstance(network, nn.DataParallel):
                    network = network.module

                # network = nn.DataParallel(network).cuda()
                ckpt = torch.load(gen_path)
                current_state_dict = network.state_dict()
                for name, param in ckpt.items():
                    if name in skip_para:
                        continue
                        # print(name)
                        # import pdb; pdb.set_trace()
                    else:
                        current_state_dict[name] = param

                network.load_state_dict(current_state_dict, strict=False)
            if self.opt['phase'] == 'train':
                self.begin_step = 0
                self.begin_epoch = 0




class DDPM_PD(BaseModel):
    def __init__(self, opt):
        super(DDPM_PD, self).__init__(opt)

        if opt['dist']:
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        # define network and load pretrained models
        self.netG_t = self.set_device(networks.define_G(opt, student=False))
        if  opt['CD'] :
            self.netG_s = self.set_device(networks.define_G(opt, student=False))
        else:
            self.netG_s = self.set_device(networks.define_G(opt, student=True))
        if opt['dist']:
            self.netG_t.to(device)
            self.netG_s.to(device)
       
        # self.netG.to(device)
       

        self.schedule_phase = None
        self.opt = opt

        # set loss and load resume state

        self.set_loss()
        self.lpips = lpips.LPIPS(net='vgg').cuda()

        # self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

        if self.opt['phase'] == 'train':
            self.netG_s.train()
            # find the parameters to optimize

            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG_s.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG_s.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            self.scheduler = get_scheduler(self.optG, opt)
            self.log_dict = OrderedDict()


       

        if self.opt['phase'] == 'test':
            self.netG_s.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
        else:
            self.load_network()
            # self.netG_t.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
            if opt['dist']:
                self.netG_s = DDP(self.netG_s, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)
                self.netG_t = DDP(self.netG_t, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)
        for p in self.netG_t.parameters():
            p.requires_grad_(False)
        self.netG_t.eval()
        self.netG_t.CD = opt['CD'] 
        self.netG_s.CD = opt['CD'] 

        # self.print_network()

        # define ema
        self.ema_decay = opt['train']["ema_scheduler"]["ema_decay"]
        if self.opt['dist']:
            self.ema_student = EMA(
                self.netG_s.module,
                decay = self.ema_decay,              # exponential moving average factor
            )
        else:
            self.ema_student = EMA(
                self.netG_s,
                decay = self.ema_decay,              # exponential moving average factor
            )
        
        self.ema_student.register()

    def feed_data(self, data):

        dic = {}

        if self.opt['dist']:
            dic = {}
            dic['LQ'] = data['LQ'].to(self.local_rank)
            dic['GT'] = data['GT'].to(self.local_rank)
            self.data = dic
        else:
            dic['LQ'] = data['LQ']
            dic['GT'] = data['GT']

            self.data = self.set_device(dic)

    def optimize_parameters(self):

        self.optG.zero_grad()
        if self.opt['dist']:
            l_pd = self.netG_t(self.data, self.netG_s.module, lpips_func=self.lpips)
        else:
            l_pd = self.netG_t(self.data, self.netG_s, lpips_func=self.lpips)
        # print("to be debug")
        # import pdb; pdb.set_trace()
        #

       
        loss = l_pd
        # print(l_pd)
        # import pdb; pdb.set_trace()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG_s.parameters(), 1)
        self.optG.step()
        self.scheduler.step()
        # print( self.optG.param_groups[0]['lr'])
        self.ema_student.update()
        # set log
        self.log_dict['total_loss'] = loss.item()

        


    def test(self, continous=False, stride=1):
        self.ema_student.apply_shadow() # apply shadow weights here
        self.netG_s.eval()
        with torch.no_grad():
            if isinstance(self.netG_s, nn.DataParallel):
                self.SR = self.netG_s.module.super_resolution(
                    self.data['LQ'], continous, stride)
                
            else:
                if self.opt['dist']:
                    self.SR = self.netG_s.module.super_resolution(self.data['LQ'], continous, stride)
                else:
                    self.SR = self.netG_s.super_resolution(self.data['LQ'], continous, stride)
        self.ema_student.restore()# restore shadow weights here

        self.netG_s.train()

    def sample(self, batch_size=1, continous=False):
        self.ema_student.apply_shadow() # apply shadow weights here
        self.netG_s.eval()
        with torch.no_grad():
            if isinstance(self.netG_s, nn.DataParallel):
                self.SR = self.netG_s.module.sample(batch_size, continous)
            else:
                self.SR = self.netG_s.sample(batch_size, continous)
        self.ema_student.restore()# restore shadow weights here
        self.netG_s.train()

    def set_loss(self):
        if isinstance(self.netG_s, nn.DataParallel):
            self.netG_s.module.set_loss(self.device)
        else:
            self.netG_s.set_loss(self.device)
    

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['HQ'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['LQ'].detach().float().cpu()
            out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
            else:
                out_dict['LQ'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG_s)
        if isinstance(self.netG_s, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG_s.__class__.__name__,
                                             self.netG_s.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG_s.__class__.__name__)

        logger.info(s)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))

    def save_network(self, distill_step, epoch, iter_step, psnr, ssim, lpips):
        save_root = os.path.join(self.opt['path']['checkpoint'], 'num_step_{}'.format(distill_step))
        os.makedirs(save_root, exist_ok=True)
        # gen_path = os.path.join(save_root, 'P{:.4e}_S{:.4e}_I{}_E{}_gen.pth'.format(psnr, ssim, iter_step, epoch))
        # opt_path = os.path.join(save_root, 'P{:.4e}_S{:.4e}_I{}_E{}_opt.pth'.format(psnr, ssim, iter_step, epoch))
        ema_path = os.path.join(save_root, 'psnr{:.4f}_ssim{:.4f}_lpips{:.4f}_I{}_E{}_gen_ema.pth'.format(psnr, ssim, lpips, iter_step, epoch))
        
        # gen
        # network = self.netG_s
        # if isinstance(self.netG_s, nn.DataParallel):
        #     network = network.module
        # state_dict = network.state_dict()
        # for key, param in state_dict.items():
        #     state_dict[key] = param.cpu()
        # torch.save(state_dict, gen_path)

        # opt
 


        # ema
        self.ema_student.apply_shadow()
        network = self.ema_student.model
        if isinstance(self.ema_student.model, nn.DataParallel):
            network = network.module
        ema_ckpt = network.state_dict()
        for key, param in ema_ckpt.items():
            ema_ckpt[key] = param.cpu()
        torch.save(ema_ckpt, ema_path)
        self.ema_student.restore()
        # logger.info(
        #     'Saved model in [{:s}] ...'.format(gen_path))
        logger.info(
            'Saved model in [{:s}] ...'.format(ema_path))
        return ema_path # gen_path

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}'.format(load_path)
            
            # gen
            networks = [self.netG_t, self.netG_s]
            for network in networks:
                if isinstance(network, nn.DataParallel):
                    network = network.module
                ckpt = torch.load(gen_path)

                current_state_dict = network.state_dict()
                for name, param in ckpt.items():
                    if name in skip_para:
                        continue

                    else:
                        current_state_dict[name] = param

                network.load_state_dict(current_state_dict, strict=False)

            if self.opt['phase'] == 'train':

                self.begin_step = 0
                self.begin_epoch = 0
