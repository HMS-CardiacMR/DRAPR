import os
import sys
import scipy.io as sio
import numpy as np

from recon import nufft
from data.twix_datasets import DatasetTwixRadial


data = DatasetTwixRadial(opt=None)

def _run_recon(twix_dir, twix_fname, device='cuda'):
    
    twix_filename = os.path.join(twix_dir, twix_fname)

    kspace_data   = data.load_twix(twix_filename, dtype=np.complex64)

    n_readout_points, n_lines, n_frames, n_slices, n_coils = kspace_data.shape

    print('Running NUFFT on kspace of shape', (n_readout_points, n_lines, n_frames, n_slices, n_coils))

    recon = nufft.NUFFT_prototype(kspace_data, device=device, numpoints=4, remove_n_time_frames=0)

    sio.savemat(os.path.join(twix_dir, twix_filename.strip('.dat')+'_reconstructed.mat'), {'recon_nufft':recon.transpose((2,3,1,0))[None, None]}) 

if __name__ == "__main__":

    twix_dir   = sys.argv[1]
    twix_fname = sys.argv[2]

    _run_recon(twix_dir, twix_fname)

