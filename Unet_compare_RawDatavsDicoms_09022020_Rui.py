import matplotlib.pyplot as plt
import sys

sys.path.append('/data1/HassanData/DeepLearning/BigDataCine/TrainingData/')
import torch
import numpy as np

## Defining the network
Hidden_layer = 64
Conv_kernel = 3
Conv_kerenl_time = 3
Padd_space = 1
Padd_time = 1
drop_out_level = 0.15
Bias = True
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        #down layer 1
        self.conv_DL1 = torch.nn.Sequential()
        self.conv_DL1.add_module("Conv_DL1",nn.Conv3d(1,Hidden_layer,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_DL1.add_module("BN1_DL1",nn.BatchNorm3d(Hidden_layer))
        self.DropOut1 = nn.Dropout3d(p=drop_out_level,inplace=True)


        self.conv_DL1_v2 = torch.nn.Sequential()
        self.conv_DL1_v2.add_module("Conv_DL1_v2",nn.Conv3d(Hidden_layer,Hidden_layer,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_DL1_v2.add_module("BN1_DL1_v2",nn.BatchNorm3d(Hidden_layer))

        self.DropOut2 = nn.Dropout3d(p=drop_out_level,inplace=True)

        # max pooling layer
        self.conv_MP1 = torch.nn.Sequential()
        self.conv_MP1.add_module("Max Pool 1",nn.MaxPool3d((2,2,2),stride = (2,2,2)))

        #down layer 2
        self.conv_DL2 = torch.nn.Sequential()
        self.conv_DL2.add_module("Conv_DL2",nn.Conv3d(Hidden_layer,Hidden_layer*2,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_DL2.add_module("BN1_DL2",nn.BatchNorm3d(Hidden_layer*2))

        self.DropOut3 = nn.Dropout3d(p=drop_out_level,inplace=True)

        self.conv_DL2_v2 = torch.nn.Sequential()
        self.conv_DL2_v2.add_module("Conv_DL2_v2",nn.Conv3d(Hidden_layer*2,Hidden_layer*2,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_DL2_v2.add_module("BN1_DL2_v2",nn.BatchNorm3d(Hidden_layer*2))

        self.DropOut4 = nn.Dropout3d(p=drop_out_level,inplace=True)

        # max pooling layer
        self.conv_MP2 = torch.nn.Sequential()
        self.conv_MP2.add_module("Max Pool 2",nn.MaxPool3d((2,2,2),stride = (2,2,2)))

        #down layer 2
        self.conv_DL3 = torch.nn.Sequential()
        self.conv_DL3.add_module("Conv_DL3",nn.Conv3d(Hidden_layer*2,Hidden_layer*4,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_DL3.add_module("BN1_DL3",nn.BatchNorm3d(Hidden_layer*4))

        self.DropOut5 = nn.Dropout3d(p=drop_out_level,inplace=True)

        self.conv_DL3_v2 = torch.nn.Sequential()
        self.conv_DL3_v2.add_module("Conv_DL3_v2",nn.Conv3d(Hidden_layer*4,Hidden_layer*4,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_DL3_v2.add_module("BN1_DL3_v2",nn.BatchNorm3d(Hidden_layer*4))

        self.DropOut6 = nn.Dropout3d(p=drop_out_level,inplace=True)

        # Conv Transpose
        self.convT1 = nn.ConvTranspose3d(Hidden_layer*4,Hidden_layer*2,(2,2,2),stride = (2,2,2))

        #up layer 1
        self.conv_UP1 = torch.nn.Sequential()
        self.conv_UP1.add_module("Conv_UP1",nn.Conv3d(Hidden_layer*4,Hidden_layer*2,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_UP1.add_module("BN1_UP1",nn.BatchNorm3d(Hidden_layer*2))

        self.DropOut7 = nn.Dropout3d(p=drop_out_level,inplace=True)

        self.conv_UP1_v2 = torch.nn.Sequential()
        self.conv_UP1_v2.add_module("Conv_UP1_v2",nn.Conv3d(Hidden_layer*2,Hidden_layer*2,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_UP1_v2.add_module("BN1_UP1_v2",nn.BatchNorm3d(Hidden_layer*2))

        self.DropOut8 = nn.Dropout3d(p=drop_out_level,inplace=True)

        # Conv Transpose
        self.convT2 = nn.ConvTranspose3d(Hidden_layer*2,Hidden_layer,(2,2,2),stride = (2,2,2))

        #up layer 2
        self.conv_UP2 = torch.nn.Sequential()
        self.conv_UP2.add_module("Conv_UP2",nn.Conv3d(Hidden_layer*2,Hidden_layer,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_UP2.add_module("BN1_UP2",nn.BatchNorm3d(Hidden_layer))

        self.DropOut9 = nn.Dropout3d(p=drop_out_level,inplace=True)

        self.conv_UP2_v2 = torch.nn.Sequential()
        self.conv_UP2_v2.add_module("Conv_UP2_v2",nn.Conv3d(Hidden_layer,Hidden_layer,(Conv_kernel,Conv_kernel,Conv_kerenl_time), padding = (Padd_space,Padd_space,Padd_time), stride = 1,bias = Bias))
        self.conv_UP2_v2.add_module("BN1_UP2_v2",nn.BatchNorm3d(Hidden_layer))

        self.DropOut10 = nn.Dropout3d(p=drop_out_level,inplace=True)

        #Final layer
        self.conv_final = torch.nn.Sequential()
        self.conv_final.add_module("Conv Final", nn.Conv3d(Hidden_layer,1,(1,1,1),padding = (0,0,0),stride = 1,bias = Bias))



    def forward(self,x):
        x_down1 = F.relu(self.DropOut1(self.conv_DL1.forward(x)))
        x_down1_v2 = F.relu(self.DropOut2(self.conv_DL1_v2.forward(x_down1)))

        x_MaxPool = self.conv_MP1.forward(x_down1_v2)

        x_down2 = F.relu(self.DropOut3(self.conv_DL2.forward(x_MaxPool)))
        x_down2_v2 = F.relu(self.DropOut4(self.conv_DL2_v2.forward(x_down2)))

        x_MaxPool_v2 = self.conv_MP2.forward(x_down2_v2)

        x_down3 = F.relu(self.DropOut5(self.conv_DL3.forward(x_MaxPool_v2)))
        x_down3_v2 = F.relu(self.DropOut6(self.conv_DL3_v2.forward(x_down3)))

        x_up1_ConvT = self.convT1(x_down3_v2,output_size = x_down2_v2.size())
        x_down2_up1_stack = torch.cat((x_down2_v2,x_up1_ConvT),1)

        x_up1 =  F.relu(self.DropOut7(self.conv_UP1.forward(x_down2_up1_stack)))
        x_up1_v2 =  F.relu(self.DropOut8(self.conv_UP1_v2.forward(x_up1)))

        x_up2_ConvT = self.convT2(x_up1_v2,output_size = x_down1_v2.size())
        x_down1_up2_stack = torch.cat((x_down1_v2,x_up2_ConvT),1)

        x_up2 = F.relu(self.DropOut9(self.conv_UP2.forward(x_down1_up2_stack)))
        x_up2_v2 = F.relu(self.DropOut10(self.conv_UP2_v2.forward(x_up2)))

        output = x+self.conv_final.forward(x_up2_v2)

        return output

## loading the trained model
PATH = "./models/model.py"
net = Net()
device2 = torch.device("cuda:5")
net = nn.DataParallel(net, device_ids=[5,6])
net.load_state_dict(torch.load(PATH))
net.to(device2)
net.eval()

##  feeding in the prospectively acquired real-time cine and reconstructing this dataset
main_file =  "./data/"
SubFile = "Prospective"
import ProspectiveRadialSaveConcatV4

# test case used
data_images_GPU = ProspectiveRadialSaveConcatV4.Save_Cases(main_file, SubFile, normalize_window = 48, Crop_nx = 144, net2 = net, device = device2)
input_all_GPU= data_images_GPU.data[0]
outp_net_GPU= data_images_GPU.data[1]
outp_all_GPU= data_images_GPU.data[2]

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 5))
ax[0].imshow(np.abs(input_all_GPU[ :, :, 16,7]), cmap='gray', vmin=0, vmax=1.75)
ax[1].imshow(np.abs(outp_net_GPU[ :, :, 16,7]), cmap='gray', vmin=0, vmax=1.75)
ax[2].imshow(np.abs(outp_all_GPU[ :, :, 16,7]), cmap='gray', vmin=0, vmax=1.75)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 5))
ax[0].imshow(np.abs(input_all_GPU[ :, :, 15,5]), cmap='gray', vmin=0, vmax=1.75)
ax[1].imshow(np.abs(outp_net_GPU[ :, :, 15,5]), cmap='gray', vmin=0, vmax=1.75)
ax[2].imshow(np.abs(outp_all_GPU[ :, :, 15,5]), cmap='gray', vmin=0, vmax=1.75)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 5))
ax[0].imshow(np.abs(input_all_GPU[ :, :, 20,5]), cmap='gray', vmin=0, vmax=1.75)
ax[1].imshow(np.abs(outp_net_GPU[ :, :, 20,5]), cmap='gray', vmin=0, vmax=1.75)
ax[2].imshow(np.abs(outp_all_GPU[ :, :, 20,5]), cmap='gray', vmin=0, vmax=1.75)

