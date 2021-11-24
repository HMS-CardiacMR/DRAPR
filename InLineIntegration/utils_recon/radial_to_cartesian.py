# Manuel A. Morales (mmorale5@bidmc.harvard.edu)
# Postdoctoral Research Fellow
# Beth Israel Deaconess Medical Center
# Cardiovascular MR Center 

import numpy as np

polar2cart = lambda r, phi : (r * np.cos(phi), r * np.sin(phi))

def radial_to_cartesian_slice_coordinates(kspace_data_shape, N=5, remove_n_time_frames=0,pi_norm=False):
    """Convert radial trajectory to Cartesian coordinates. 

    Inputs
    ------
    kspace_data_shape : tuple of integers, (n_readout_points, n_lines, n_time_frames)

    N : integer, angle sequence element (see DOI: 10.1109/TMI.2014.2382572).
         N = 1   golden angle
         N = 2   complementary small golden angle
         N > 2   tiny golden angles

    remove_n_time_frames : integer, the time frame at which acquisition was started. If > 0, the
                           resulting trajectory assumes the first `remove_n_time_frames` frames
                           will be removed from the corresponding k-space data.

    pi_norm : torch-based NUFFT implementation scales coordinates from -pi to pi.

    Return
    ------

    x, y : the Cartesian coordinates of the acquisition trajectory, each with shape (n_readout_points, n_lines, n_time_frames).

    """

    n_readout_points, n_lines, n_time_frames = kspace_data_shape

    tau = (1 + 5**0.5) / 2.
    golden_angle = np.pi / (tau + N - 1)
    remove_n_lines = n_lines * remove_n_time_frames
    angle_offset = golden_angle * remove_n_lines

    if pi_norm:
        rho = np.arange(-0.5, 0.5, 1.0/n_readout_points) * 2 * np.pi
    else:
        rho = np.arange(-(n_readout_points/2.-1), n_readout_points/2.+1)

    phi = np.arange(0, n_lines * n_time_frames) * golden_angle + angle_offset
    phi = np.mod(phi, np.pi)

    # indexing='ij' and order='F' is needed to match older Matlab implementation.
    rho, phi = np.meshgrid(rho, phi, indexing='ij')
    x, y = polar2cart(rho, phi)

    x = x.reshape((n_readout_points, n_lines, n_time_frames), order='F')
    y = y.reshape((n_readout_points, n_lines, n_time_frames), order='F')

    return x, y