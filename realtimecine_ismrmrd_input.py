import ismrmrd
import numpy as np

from scipy import io

class feed_to_network(object):

    def __init__(self, main_file,SubFile,normalize_window = 48,Crop_nx = 144,net2 = None,device = None):
        self.search_path_zp = main_file + SubFile + "/Gridded Data/"
        self.search_path_save_NET = main_file + SubFile + "/NN Output/"
        self.net2 = net2
        self.device = device
        self.normalize_window = normalize_window
        self.Crop_nx = Crop_nx
        self.data = self._save_files()

    def _save_files(self):

        # Create target Directory
        try:
          os.makedirs(self.search_path_save_NET)
          print("Directory ", self.search_path_save_NET, " Created ")
        except FileExistsError:
          print("Directory ", self.search_path_save_NET, " already exists")

          # Load ISMRMRD formmated data
          dataset = ismrmrd.Dataset('data/ISMRMRD/MRD_input_2021-07-22-095809_86.h5', '/dataset', True)

          # Count total images
          nImages = dataset.number_of_images('images_0')

          # Create array that will be populated with image date
          mat_zp2 = np.zeros((1,144,144,nImages,1), dtype='Complex32')

          # Populate array with data from image
          for img_count in range(nImages):
            image = dataset.read_image('images_0', img_count)
            mat_zp2[0,:,:,img_count,0] = image.data[0,0,...]

          # Select start and end index based on the crop
          startx1 = np.floor(mat_zp2.shape[1] / 2 - self.Crop_nx / 2).astype(int)
          endx1 = np.floor(mat_zp2.shape[1] / 2 + self.Crop_nx / 2).astype(int)
          starty1 = np.floor(mat_zp2.shape[2] / 2 - self.Crop_nx / 2).astype(int)
          endy1 = np.floor(mat_zp2.shape[2] / 2 + self.Crop_nx / 2).astype(int)
          mat_zp = mat_zp2[:, startx1:endx1, starty1:endy1, :, :]

          # Create empty output arrays
          outp_net = np.zeros([mat_zp.shape[1], mat_zp.shape[2], mat_zp.shape[3], mat_zp.shape[4]], dtype='Complex32')

          # Loop through all the slices
          for zz in range(0, mat_zp.shape[4]):

              startx1 = np.floor(mat_zp2.shape[1] / 2 - self.Crop_nx / 2).astype(int)
              endx1 = np.floor(mat_zp2.shape[1] / 2 + self.Crop_nx / 2).astype(int)
              starty1 = np.floor(mat_zp2.shape[2] / 2 - self.Crop_nx / 2).astype(int)
              endy1 = np.floor(mat_zp2.shape[2] / 2 + self.Crop_nx / 2).astype(int)
              mat_zp = (mat_zp2[:, startx1:endx1, starty1:endy1, :, zz])

              # Apply normalization?
              nx_crop2 = self.normalize_window
              startx = np.floor(mat_zp.shape[1] / 2 - nx_crop2 / 2).astype(int)
              endx = np.floor(mat_zp.shape[1] / 2 + nx_crop2 / 2).astype(int)
              mat_zp_crop = mat_zp[:, startx:endx, startx:endx, :]
              mat_zp = mat_zp / np.percentile(np.abs(mat_zp_crop), 95)

              # Create input and output arrays
              inpt = np.zeros([1, 1, mat_zp.shape[1]*2, mat_zp.shape[2], mat_zp.shape[3]], dtype='float32')

              # Storing real and imaginary parts
              inpt[0, 0, 0:mat_zp.shape[2], :, :] = np.real(mat_zp[0, :, :, :])
              inpt[0, 0, mat_zp.shape[2]:mat_zp.shape[2]*2, :, :] = np.imag(mat_zp[0, :, :, :])

              # Feed data into the network
              inputs = torch.from_numpy(inpt)
              inputs = inputs.to(self.device)
              outputs = self.net2(inputs)
              outputs = outputs.cpu()
              outputs = outputs.data.numpy()

              # Selecting subset of data to save
              outputs2 = np.abs(outputs[:, :, 0:mat_zp.shape[1], :, :] + 1j * outputs[:, :,mat_zp.shape[1]:mat_zp.shape[1]*2, :, :])

              outp_net[:, :, :, zz] = outputs2[0, 0, :, :, :]

              print('Completed Slice:', zz)

              io.savemat(str(self.search_path_save_NET + filename_zp), {'outp_net': outp_net})

          return outp_net
      
      def __call__(self):
        outp_net = self._save_files()
        return outp_net