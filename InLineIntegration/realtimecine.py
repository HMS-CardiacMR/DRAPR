import ismrmrd
import os
import itertools
import logging
import numpy as np
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import perf_counter
from network_arch import Net
from scipy import io

import nufft

debugFolder = "/tmp/share/debug"  # Folder for debug output files
use_gpu = True                    # Enable/Disable GPU Use
n_threads = 12                    # Set number of threads for PyTorch
frame_skip = 20                   # Frames to skip to reach steady state

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    imgGroup = []
    kdataGroup = []

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):
                imgGroup.append(item)

            # ----------------------------------------------------------
            # Raw k-space data
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Acquisition):
                kdataGroup.append(item)


        if len(imgGroup) > 0:
            logging.info("Processing a group of images")
            cine_movies, head = process_image(imgGroup, connection, config, metadata)

        if len(kdataGroup) > 0:
            logging.info("Processing a group of k-space data")
            images = process_kspace(kdataGroup, connection, config, metadata)
            logging.info("Processing pre-processed images")
            cine_movies = process_image(images, connection, config, metadata)

        # logging.info('Cine movies have shape: %s', str(cine_movies.shape))
        # io.savemat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', 'cine_movies.mat'), {'outp_net': cine_movies})

        for slice_index in range(cine_movies.shape[-1]):
            
            for image_index in range(cine_movies[..., slice_index].shape[-1]):
                
                data = cine_movies[..., image_index, slice_index]
                data *= 1000
                data = data.astype(np.int16)

                data = np.rot90(data, 3) # Rotate clockwise 90 degrees
                
                # Format as ISMRMRD image data
                image = ismrmrd.Image.from_array(data)

                # # Set the header information
                # tmpHead = head[slice_index][image_index]
                # tmpHead.data_type = image.getHead().data_type
                # tmpHead.image_index = image_index
                # tmpHead.image_series_index = 0
                # image.setHead(tmpHead)

                # # Set field of view
                # image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                #                         ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                #                         ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

                # # Create a copy of the original ISMRMRD Meta attributes and update
                # tmpMeta = ismrmrd.Meta.deserialize(image.attribute_string)
                # tmpMeta['DataRole']                       = 'Image' 
                # tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'REALTIMECINE']
                # tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'
                # tmpMeta['WindowCenter']                   = '1500'
                # tmpMeta['WindowWidth']                    = '4000'

                # # Add image orientation directions to MetaAttributes if not already present
                # if tmpMeta.get('ImageRowDir') is None:
                #     tmpMeta['ImageRowDir'] = ["{:.18f}".format(tmpHead.read_dir[0]), "{:.18f}".format(tmpHead.read_dir[1]), "{:.18f}".format(tmpHead.read_dir[2])]

                # if tmpMeta.get('ImageColumnDir') is None:
                #     tmpMeta['ImageColumnDir'] = ["{:.18f}".format(tmpHead.phase_dir[0]), "{:.18f}".format(tmpHead.phase_dir[1]), "{:.18f}".format(tmpHead.phase_dir[2])]

                # metaXml = tmpMeta.serialize()

                # logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
                # logging.debug("Image data has %d elements", image.data.size)

                # image.attribute_string = metaXml

                # Send image back to the client
                logging.debug("Sending images to client")
                connection.send_image(image)

    finally:
        connection.send_close()

def process_kspace(kspace, connection, config, metadata):

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    
    n_readout_points = kspace[-1].data.shape[1]
    n_coils   = kspace[-1].data.shape[0]
    # Following values are an index so we must increment by 1
    n_slices  = kspace[-1].idx.slice + 1
    n_frames  = kspace[-1].idx.phase + 1
    n_lines   = kspace[-1].idx.kspace_encode_step_1 + 1

    data = np.stack([sample.data for sample in kspace])             # (total_samples_collected, n_coils, n_readout_points)

    data_reshaped = np.reshape(data, (n_lines,n_frames, n_slices, n_coils, n_readout_points), order='F')

    data_transposed = data_reshaped.transpose((4, 0, 1, 2, 3))      # (n_readout_points, n_lines, n_frames, n_slices, n_coils)

    image_recon_combined = nufft.NUFFT(data_transposed, device='cuda', remove_n_time_frames=frame_skip)

    return image_recon_combined

def process_image(images, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # (n_slice, n_phase, x, y) -> (x, y, n_phase, n_slice)
    data = data.transpose((3, 2, 1, 0))
    # Adding z-axis
    data = data[np.newaxis, ...]

    # logging.info("Saving input to the network.")
    # io.savemat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', 'input_to_network.mat'), {'input_net': data[0,:,:,:,:]})

    logging.info("Incoming dataset contains %s encodings", str(data.shape))

    normalize_window = 48
    crop_nx = 144

    # Select start and end index based on the crop
    startx1 = np.floor(data.shape[1] / 2 - crop_nx / 2).astype(int)
    endx1 = np.floor(data.shape[1] / 2 + crop_nx / 2).astype(int)
    starty1 = np.floor(data.shape[2] / 2 - crop_nx / 2).astype(int)
    endy1 = np.floor(data.shape[2] / 2 + crop_nx / 2).astype(int)
    mat_zp = data[:, startx1:endx1, starty1:endy1, frame_skip:, :]

    # Create empty output arrays
    outp_net = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype='Complex32')

    # Loop through all the slices
    for zz in range(0, mat_zp.shape[4]):

      startx1 = np.floor(data.shape[1] / 2 - crop_nx / 2).astype(int)
      endx1 = np.floor(data.shape[1] / 2 + crop_nx / 2).astype(int)
      starty1 = np.floor(data.shape[2] / 2 - crop_nx / 2).astype(int)
      endy1 = np.floor(data.shape[2] / 2 + crop_nx / 2).astype(int)
      mat_zp = (data[:, startx1:endx1, starty1:endy1, frame_skip:, zz])

      # Apply normalization
      startx = np.floor(mat_zp.shape[1] / 2 - normalize_window / 2).astype(int)
      endx = np.floor(mat_zp.shape[1] / 2 + normalize_window / 2).astype(int)
      mat_zp_crop = mat_zp[:, startx:endx, startx:endx, :]
      mat_zp = mat_zp / np.percentile(np.abs(mat_zp_crop), 95)

      # Create input and output arrays
      inpt = np.zeros([1, 1, mat_zp.shape[1]*2, mat_zp.shape[2], mat_zp.shape[3]], dtype='float32')

      # Storing real and imaginary parts
      inpt[0, 0, 0:mat_zp.shape[2], :, :] = np.real(mat_zp[0, :, :, :])
      inpt[0, 0, mat_zp.shape[2]:mat_zp.shape[2]*2, :, :] = np.imag(mat_zp[0, :, :, :])

      torch.set_num_threads(n_threads)

      if use_gpu:
        logging.info("Using GPU")
        ## loading the trained model
        PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'model.py')
        net = Net()
        device = torch.device("cuda:0")
        net = nn.DataParallel(net, device_ids=[0])
        net.load_state_dict(torch.load(PATH))
        net = net.to(device)
        net.eval()

        # Feed data into the network
        with torch.no_grad():
          inputs = torch.from_numpy(inpt)
          inputs = inputs.to(device)
          outputs = net(inputs)
          outputs = outputs.cpu()
          outputs = outputs.data.numpy()

      else:
        logging.info("Using CPU")
        ## loading the trained model
        PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'model.py')
        net = Net()
        device = torch.device("cpu")
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(PATH, map_location="cpu"))
        net = net.module.to(device)
        net.eval()

        # Feed data into the network
        with torch.no_grad():
          inputs = torch.from_numpy(inpt)
          inputs = inputs.to(device)
          outputs = net(inputs)
          outputs = outputs.cpu()
          outputs = outputs.data.numpy()

      # Selecting subset of data to save
      outputs2 = np.abs(outputs[:, :, 0:mat_zp.shape[1], :, :] + 1j * outputs[:, :,mat_zp.shape[1]:mat_zp.shape[1]*2, :, :])

      outp_net[:, :, :, zz] = outputs2[0, 0, :, :, :]

      logging.info("Completed Slice: %d", zz)

    return outp_net
