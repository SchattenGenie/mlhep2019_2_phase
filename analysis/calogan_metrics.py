import numpy as np
def get_assymetry(imgs, ps, points, orthog=False):
    assym_res = []
    zoff = 25
    
    x = np.linspace(-14.5, 14.5, 30)
    y = np.linspace(-14.5, 14.5, 30)
    xx, yy = np.meshgrid(x, y)
    xx = np.repeat(xx[np.newaxis, ...], len(imgs), axis=0)
    yy = np.repeat(yy[np.newaxis, ...], len(imgs), axis=0)
    
    points_0 = points[:, 0] + zoff * ps[:, 0] / ps[:, 2]
    points_1 = points[:, 1] + zoff * ps[:, 1] / ps[:, 2]
    if orthog:
        line_func = lambda x: (x - points_0[..., np.newaxis, np.newaxis]) / (ps[:, 0] / ps[:, 1])[..., np.newaxis, np.newaxis] + points_1[..., np.newaxis, np.newaxis]
    else:
        line_func = lambda x: -(x - points_0[..., np.newaxis, np.newaxis]) / (ps[:, 1] / ps[:, 0])[..., np.newaxis, np.newaxis] + points_1[..., np.newaxis, np.newaxis]

    sign = np.ones(len(ps))
    if not orthog:
        sign = (ps[:, 1] > 0).astype(int)
        sign = 2 * (sign - 0.5)
    
    idx = np.where((yy - line_func(xx)) * sign[..., np.newaxis, np.newaxis] < 0)        
    zz = np.ones((len(imgs), 30, 30))
    zz[idx] = 0
    assym = (np.sum(imgs * zz, axis=(1, 2)) - 
             np.sum(imgs * (1 - zz), axis=(1, 2))) / np.sum(imgs, axis=(1, 2))
        
    return assym

def zz_to_line(zz):
    res = (
        np.concatenate([np.abs(np.diff(zz, axis=2)), np.zeros((len(zz), 30, 1))], axis=2) + 
        np.concatenate([np.abs(np.diff(zz, axis=1)), np.zeros((len(zz), 1, 30))], axis=1)
    )
    return np.clip(res, 0, 1)

def get_shower_width(data, ps, points, orthog=False):
    res = []
    spreads = []
    
    assym_res = []
    zoff = 25
    
    x = np.linspace(-14.5, 14.5, 30)
    y = np.linspace(-14.5, 14.5, 30)
    xx, yy = np.meshgrid(x, y)
    xx = np.repeat(xx[np.newaxis, ...], len(data), axis=0)
    yy = np.repeat(yy[np.newaxis, ...], len(data), axis=0)
    
    points_0 = points[:, 0] + zoff * ps[:, 0] / ps[:, 2]
    points_1 = points[:, 1] + zoff * ps[:, 1] / ps[:, 2]
    
    if orthog:
        line_func = lambda x: -(x - points_0[..., np.newaxis, np.newaxis]) / (ps[:, 0] / ps[:, 1])[..., np.newaxis, np.newaxis] + points_1[..., np.newaxis, np.newaxis]
    else:
        line_func = lambda x: (x - points_0[..., np.newaxis, np.newaxis]) / (ps[:, 1] / ps[:, 0])[..., np.newaxis, np.newaxis] + points_1[..., np.newaxis, np.newaxis]
    rescale = np.sqrt(1 + (ps[:, 1] / ps[:, 0])**2)
    

    sign = np.ones(len(ps))
    if not orthog:
        sign = (ps[:, 1] < 0).astype(int)
        sign = 2 * (sign - 0.5)
    
    idx = np.where((yy - line_func(xx)) * sign[..., np.newaxis, np.newaxis] < 0)        
    zz = np.ones((len(data), 30, 30))
    zz[idx] = 0
    line = zz_to_line(zz)
    
    ww = (line * data) # * rescale[..., np.newaxis, np.newaxis]
    sum_0 = ww.sum(axis=(1, 2))
    sum_1 = (ww * rescale[..., np.newaxis, np.newaxis] * xx).sum(axis=(1, 2))
    sum_2 = (ww * (rescale[..., np.newaxis, np.newaxis] * xx)**2).sum(axis=(1, 2))
    
    sum_1 = sum_1 / sum_0
    sum_2 = sum_2 / sum_0

    sigma = np.sqrt(sum_2 - sum_1 * sum_1)
        
    return sigma


def get_ms_ratio2(data, alpha=0.1):
    ms = np.sum(data, axis=(1, 2))
    num = np.sum((data >= (ms * alpha)[:, np.newaxis, np.newaxis]), axis=(1, 2))
    return num / 900.

def get_sparsity_level(data):
    alphas = np.logspace(-5, 0, 50)
    sparsity = []
    for alpha in alphas:
        sparsity.append(get_ms_ratio2(data, alpha))
    return np.array(sparsity)


def get_physical_stats(EnergyDeposit, ParticleMomentum, ParticlePoint):
    assym = get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)
    assym_ortho = get_assymetry(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)
    sh_width = get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=False)
    sh_width_ortho = get_shower_width(EnergyDeposit, ParticleMomentum, ParticlePoint, orthog=True)
    sparsity_level = get_sparsity_level(EnergyDeposit)
    stats = np.c_[
        assym, 
        assym_ortho, 
        sh_width, 
        sh_width_ortho, 
        sparsity_level.T
    ]
    return stats
