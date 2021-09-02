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

# Folder for debug output files
debugFolder = "/tmp/share/debug"

# Enable/Disable GPU Use
use_gpu = False

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

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):

                imgGroup.append(item)

        if len(imgGroup) > 0:
            logging.info("Processing a group of images")
            cine_movies = process_image(imgGroup, connection, config, metadata)

        # logging.info('Cine movies have shape: %s', str(cine_movies.shape))
        # io.savemat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', 'cine_movies.mat'), {'outp_net': cine_movies})

        for slice_index in range(cine_movies.shape[-1]):
            
            for image_index in range(cine_movies[..., slice_index].shape[-1]):
                
                data = cine_movies[..., image_index, slice_index]
                data *= 1000
                data = data.astype(np.int16)

                # Transpose the image
                data = np.rot90(data, 1) # Rotate 90 degrees
                data = np.flipud(data)   # Flip vertically
                
                # Format as ISMRMRD image data
                image = ismrmrd.Image.from_array(data)

                image.image_series_index = slice_index

                # Set field of view
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

                # Create a copy of the original ISMRMRD Meta attributes and update
                tmpMeta = ismrmrd.Meta.deserialize(image.attribute_string)
                tmpMeta['DataRole']                       = 'Image' 
                tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'REALTIMECINE']
                tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'

                metaXml = tmpMeta.serialize()

                logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
                logging.debug("Image data has %d elements", image.data.size)

                image.attribute_string = metaXml

                # Send image back to the client
                logging.debug("Sending images to client")
                connection.send_image(image)

    finally:
        connection.send_close()


def process_image(images, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Extract image data into a 4D array of size [img z y x]
    data = np.stack([img.data[0,...]                       for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    nSlices = int(images[-1].slice + 1)         # Slice is an index so we need to add 1
    nPhases = int(len(images)/nSlices)          # Equal number of phases per slice

    data = data.reshape(nSlices, nPhases, data.shape[1], data.shape[2], data.shape[3]) # slice, phase, z, y, x
    data = data.transpose((2, 4, 3, 1, 0))                                             # z, x, y, phase, slice

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
    mat_zp = data[:, startx1:endx1, starty1:endy1, :, :]

    # Create empty output arrays
    outp_net = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype='Complex32')

    # Loop through all the slices
    for zz in range(0, mat_zp.shape[4]):

      startx1 = np.floor(data.shape[1] / 2 - crop_nx / 2).astype(int)
      endx1 = np.floor(data.shape[1] / 2 + crop_nx / 2).astype(int)
      starty1 = np.floor(data.shape[2] / 2 - crop_nx / 2).astype(int)
      endy1 = np.floor(data.shape[2] / 2 + crop_nx / 2).astype(int)
      mat_zp = (data[:, startx1:endx1, starty1:endy1, :, zz])

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