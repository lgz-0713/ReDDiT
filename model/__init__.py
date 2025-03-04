import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    from .model import DDPM_PD as M_PD
    # print(opt['distill'])
    # import pdb; pdb.set_trace()
    if opt['distill']:
        m=M_PD(opt)
    else:
        m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
