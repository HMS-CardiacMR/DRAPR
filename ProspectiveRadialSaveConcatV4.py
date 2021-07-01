# import cv2
#import glob
import numpy as np
import os
import random
import hdf5storage
import torch

from scipy.interpolate import griddata
import scipy
import scipy.misc as ImResize
from PIL import Image

from skimage.transform import resize
from skimage import data

import math
import matplotlib.pyplot as plt
import time

class Save_Cases(object):

    def __init__(self, main_file,SubFile,normalize_window = 48,Crop_nx = 144,net2 = None,device = None):
        self.search_path_zp = main_file + SubFile + "/NuFFT_ZF/"
        self.search_path_rc = main_file + SubFile + "/NuFFT_RC/"
        self.search_path_save_NET = main_file + SubFile + "/NuFFT_NET_saving/"
        self.search_path_save_RC = main_file + SubFile + "/NuFFT_RC_saving/"
        self.search_path_save_ZF = main_file + SubFile + "/NuFFT_ZF_saving/"
        self.net2 = net2
        self.device = device
        self.normalize_window = normalize_window
        self.Crop_nx = Crop_nx
        self.data = self._save_files()

    def _save_files(self):

        try:
            # Create target Directory
            os.makedirs(self.search_path_save_NET)
            print("Directory ", self.search_path_save_NET, " Created ")
        except FileExistsError:
            print("Directory ", self.search_path_save_NET, " already exists")

        try:
            # Create target Directory
            os.makedirs(self.search_path_save_RC)
            print("Directory ", self.search_path_save_RC, " Created ")
        except FileExistsError:
            print("Directory ", self.search_path_save_RC, " already exists")

        try:
            # Create target Directory
            os.makedirs(self.search_path_save_ZF)
            print("Directory ", self.search_path_save_ZF, " Created ")
        except FileExistsError:
            print("Directory ", self.search_path_save_ZF, " already exists")


        ListofFiles_zp = os.listdir(self.search_path_zp)
        ListofFiles_zp = [x for x in ListofFiles_zp if os.path.isfile(os.path.join(self.search_path_zp, x))]
        print('List of files:', ListofFiles_zp)
        ns = len(ListofFiles_zp)
        start_file_index = 0
        # for jj in range(0,ns):
        for jj in range(start_file_index, ns):
            time_start = time.clock()
            ListofFiles_zp_string = str(ListofFiles_zp[jj])
            filename_zp = ListofFiles_zp_string
            print('Processing File:', filename_zp)
            load_path_zp = self.search_path_zp + filename_zp
            load_path_rc = self.search_path_rc + filename_zp

            mat_zp2 = hdf5storage.loadmat(load_path_zp)
            mat_zp2 = np.complex64(list(mat_zp2.values()))
            mat_rc2 = hdf5storage.loadmat(load_path_rc)
            mat_rc2 = np.complex64(list(mat_rc2.values()))

            startx1 = np.floor(mat_zp2.shape[1] / 2 - self.Crop_nx / 2).astype(int)
            endx1 = np.floor(mat_zp2.shape[1] / 2 + self.Crop_nx / 2).astype(int)
            starty1 = np.floor(mat_zp2.shape[2] / 2 - self.Crop_nx / 2).astype(int)
            endy1 = np.floor(mat_zp2.shape[2] / 2 + self.Crop_nx / 2).astype(int)
            mat_zp = mat_zp2[:, startx1:endx1, starty1:endy1, :, :]
            mat_rc = mat_rc2[:, startx1:endx1, starty1:endy1, :, :]

            outp_net = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype='Complex32')
            outp_all = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype='Complex32')
            input_all = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype='Complex32')

            for zz in range(0, mat_zp.shape[4]):
                #time_start = time.clock()
                startx1 = np.floor(mat_zp2.shape[1] / 2 - self.Crop_nx / 2).astype(int)
                endx1 = np.floor(mat_zp2.shape[1] / 2 + self.Crop_nx / 2).astype(int)
                starty1 = np.floor(mat_zp2.shape[2] / 2 - self.Crop_nx / 2).astype(int)
                endy1 = np.floor(mat_zp2.shape[2] / 2 + self.Crop_nx / 2).astype(int)
                mat_zp = (mat_zp2[:, startx1:endx1, starty1:endy1, :, zz])
                mat_rc = (mat_rc2[:, startx1:endx1, starty1:endy1, :, zz])
                nxx = mat_zp.shape[1]
                nx_crop2 = self.normalize_window
                startx = np.floor(mat_zp.shape[1] / 2 - nx_crop2 / 2).astype(int)
                endx = np.floor(mat_zp.shape[1] / 2 + nx_crop2 / 2).astype(int)
                mat_zp_crop = mat_zp[:, startx:endx, startx:endx, :]
                mat_zp = mat_zp / np.percentile(np.abs(mat_zp_crop), 95)
                mat_rc_crop = mat_rc[:, startx:endx, startx:endx, :]
                mat_rc = mat_rc / np.percentile(np.abs(mat_rc_crop), 95)
                inpt = np.zeros([1, 1, mat_zp.shape[1]*2, mat_zp.shape[2], mat_zp.shape[3]], dtype='float32')
                outp = np.zeros([1, 1, mat_zp.shape[1]*2, mat_zp.shape[2], mat_zp.shape[3]], dtype='float32')

                inpt[0, 0, 0:mat_zp.shape[2], :, :] = np.real(mat_zp[0, :, :, :])
                inpt[0, 0, mat_zp.shape[2]:mat_zp.shape[2]*2, :, :] = np.imag(mat_zp[0, :, :, :])

                outp[0, 0, 0:mat_zp.shape[2], :, :] = np.real(mat_rc[0, :, :, :])
                outp[0, 0, mat_zp.shape[2]:mat_zp.shape[2]*2, :, :] = np.imag(mat_rc[0, :, :, :])

                inputs = torch.from_numpy(inpt)
                inputs = inputs.to(self.device)
                outputs = self.net2(inputs)
                outputs = outputs.cpu()
                outputs = outputs.data.numpy()

                outputs2 = np.abs(outputs[:, :, 0:mat_zp.shape[1], :, :] + 1j * outputs[:, :,mat_zp.shape[1]:mat_zp.shape[1]*2, :, :])
                inputs = np.abs(inpt[:, :, 0:mat_zp.shape[1], :, :] + 1j * inpt[:, :,mat_zp.shape[1]:mat_zp.shape[1]*2, :, :])
                outp = np.abs(outp[:, :, 0:mat_zp.shape[1], :, :] + 1j * outp[:, :,mat_zp.shape[1]:mat_zp.shape[1]*2, :, :])

                outp_net[:, :, :, zz] = outputs2[0, 0, :, :, :]
                outp_all[:, :, :, zz] = outp[0, 0, :, :, :]
                input_all[:, :, :, zz] = inputs[0, 0, :, :, :]
                #print( time.clock()-time_start)

                print(zz)


            print( time.clock()-time_start)
            from scipy import io
            io.savemat(str(self.search_path_save_NET + filename_zp), {'outp_net': outp_net})
            io.savemat(str(self.search_path_save_ZF + filename_zp), {'input_all': input_all})
            io.savemat(str(self.search_path_save_RC + filename_zp), {'outp_all': outp_all})

        return input_all,outp_net,outp_all


    def __call__(self):
        input_all,outp_net,outp_all =   self._save_files()
        return input_all,outp_net,outp_all