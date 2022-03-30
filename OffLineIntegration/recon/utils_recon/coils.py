# Manuel A. Morales (mmorale5@bidmc.harvard.edu)
# Postdoctoral Research Fellow
# Beth Israel Deaconess Medical Center (BIDMC)
# Cardiovascular MR Center 

# Modified/Torch version of: 
# https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/coils.py

import torch
import numpy as np


def calculate_csm_inati_iter_prototype(im):
    """ Estimate coil map based.
    """

    im = im[:,None]
    ncha = len(im)
    D_sum = im.sum(axis=(1, 2, 3))
    v = 1/torch.linalg.norm(D_sum)
    D_sum *= v
    R = 0
    for cha in range(ncha):
        R += torch.conj(D_sum[cha]) * im[cha,...]

    eps = torch.finfo(im.real.dtype).eps * torch.abs(im).mean()
    eps.to(im)

    R = torch.conj(R)

    coil_map = im * R[None, ...]

    coil_map_conv = coil_map
    D = coil_map_conv * torch.conj(coil_map_conv)
    R = D.sum(axis=0)
    R = torch.sqrt(R) + eps
    R = 1/R
    coil_map = coil_map_conv * R[np.newaxis, ...]
    D = im * torch.conj(coil_map)
    R = D.sum(axis=0)
    D = coil_map * R[None, ...]

    D_sum = D.sum(axis=(1, 2, 3))

    v = 1/torch.linalg.norm(D_sum)
    D_sum *= v

    imT = 0
    for cha in range(ncha):
        imT += torch.conj(D_sum[cha]) * coil_map[cha, ...]

    magT = torch.abs(imT) + eps
    imT /= magT
    R = R * imT
    imT = torch.conj(imT)
    coil_map = coil_map * imT[None, ...]

    coil_map = coil_map[:, 0, :, :]

    return coil_map