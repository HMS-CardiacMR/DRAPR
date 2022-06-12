import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def process_image(images, Net, model_path, use_gpu=False, n_threads=12):
    """ Filter complex images using neural network:
        (n_slice, n_phase, x, y) -> (x, y, n_phase, n_slice)
    
    """
    data = images.transpose((3, 2, 1, 0))
    # Adding z-axis
    data = data[np.newaxis, ...]

    normalize_window = 48
    crop_nx = 144

    # Select start and end index based on the crop
    startx1 = np.floor(data.shape[1] / 2 - crop_nx / 2).astype(int)
    endx1   = np.floor(data.shape[1] / 2 + crop_nx / 2).astype(int)
    starty1 = np.floor(data.shape[2] / 2 - crop_nx / 2).astype(int)
    endy1   = np.floor(data.shape[2] / 2 + crop_nx / 2).astype(int)
    mat_zp  = data[:, startx1:endx1, starty1:endy1, :, :]

    # Create empty output arrays
    outp_net = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype=np.complex64)

    # Loop through all the slices
    for zz in range(0, mat_zp.shape[4]):

      mat_zp = (data[:, startx1:endx1, starty1:endy1, :, zz])

      # Apply normalization
      startx = np.floor(mat_zp.shape[1] / 2 - normalize_window / 2).astype(int)
      endx = np.floor(mat_zp.shape[1] / 2 + normalize_window / 2).astype(int)
      mat_zp_crop = mat_zp[:, startx:endx, startx:endx, :]
      mat_zp = mat_zp / np.percentile(np.abs(mat_zp_crop), 95)

      # Create input and output arrays
      inpt = np.zeros([1, 1, mat_zp.shape[1]*2, mat_zp.shape[2], mat_zp.shape[3]], dtype=np.complex64)

      # Storing real and imaginary parts
      inpt = np.concatenate((np.real(mat_zp), np.imag(mat_zp)), axis=1)[None].astype(np.float32)

      torch.set_num_threads(n_threads)

      if use_gpu:

        ## loading the trained model
        net = Net()
        device = torch.device("cuda:5")
        net = nn.DataParallel(net, device_ids=[5])
        net.load_state_dict(torch.load(model_path))
        net = net.to(device)
        net.eval()

        # Feed data into the network
        with torch.no_grad():
          inputs  = torch.from_numpy(inpt)
          inputs  = inputs.to(device)
          outputs = net(inputs)
          outputs = outputs.cpu()
          outputs = outputs.data.numpy()

      else:

        ## loading the trained model
        net = Net()
        device = torch.device("cpu")
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net = net.module.to(device)
        net.eval()

        # Feed data into the network
        with torch.no_grad():
          inputs = torch.from_numpy(inpt)
          #inputs = inputs.to(device)
          outputs = net(inputs)
          #outputs = outputs.cpu()
          outputs = outputs.data.numpy()


      # Selecting subset of data to save
      outputs2 = np.abs(outputs[:, :, 0:mat_zp.shape[1], :, :] + 1j * outputs[:, :,mat_zp.shape[1]:mat_zp.shape[1]*2, :, :])

      outp_net[:, :, :, zz] = outputs2[0, 0, :, :, :]

    return np.transpose(outp_net, (1,0,2,3))
