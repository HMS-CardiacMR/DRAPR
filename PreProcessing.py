# Program to perform gridding of radial data

import os
import mapvbvd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def MCNUFFT_GPU_indv(k, w, b1):
  # Multi-coil NuFFT

  return np.array([1])

def time_average(kSpaceData, nSamples, mod_switch, frames_keep):
  # Making autocalibrated b1 masks
  kSpace_ref2 = kSpaceData;
  kdata_corrected_ref = kSpaceData;

  # Getting rid of possible bad coils
  [nColumns,nLines,nPhases,nCoils] = kdata_corrected_ref.shape
  # kdata_ref = np.reshape(kdata_corrected_ref, (nColumns,nLines*nPhases,nCoils))
  kdata_ref = kdata_corrected_ref.reshape(nColumns,nLines*nPhases,nCoils)

  nx = kdata_ref.shape[0]
  tau = (1.0 + np.sqrt(5.0)) / 2.0
  golden_angle = np.pi / (tau + nSamples - 1)
  Steadystate_rays = kSpaceData.shape[1] * (frames_keep - 1)
  offSet = golden_angle * (Steadystate_rays)
  Angles = [golden_angle * i + offSet for i in range(0,nPhases*nLines)]

  if mod_switch == 1:
    Angles = np.mod(Angles,np.pi)

  ray1 = [i for i in np.arange(-0.5, 0.5, 1/nx)]

  k = np.zeros((len(ray1),len(Angles)))

  for mm in range(0,len(Angles)):
    k[:,mm] = [i*np.exp(1j*Angles[mm]) for i in ray1]

  w = np.abs(k) / max(abs(k.flatten()))
  w[np.isnan(w)] = 1
  kdata_ref = kdata_ref*np.sqrt(np.transpose(np.repeat(w[:, :, np.newaxis], nCoils, axis=2),[0,1,2])) # Why are we re-ordering with the same order?
  
  # Multicoil NUFFT operator
  E = MCNUFFT_GPU_indv(k, w, np.ones((kdata_ref.shape[0], kdata_ref.shape[0], kdata_ref.shape[2])))

  y = np.zeros((kdata_ref.shape[0]*kdata_ref.shape[1], nCoils))
  for kk in range(0,nCoils):
      y[:,kk] = kdata_ref[:,:,kk].flatten()

  rt_Cine = E.T * y

  return rt_Cine

def coil_unstreaking(kSpaceData, nSamples, nCoils, frames_keep):
    
    image_streaking_artifact = time_average(kSpaceData[:,:,0:nSamples,:], nSamples, 1, frames_keep)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def gridding(sparse_radial_data, N, timeframe):
    
    [nCol,nLin,nLineshs] = sparse_radial_data.shape;

    X = np.zeros((nCol,nLin))
    Y = np.zeros((nCol,nLin))

    grid_X = np.zeros((nCol,nLin,nLineshs))
    grid_Y = np.zeros((nCol,nLin,nLineshs))

    n_temp_X = np.zeros((nCol,nLin,nLineshs))
    n_temp_Y = np.zeros((nCol,nLin,nLineshs))

    g_cart_k_space_sparse = np.zeros((nCol,nCol,nLineshs))
    g_mask_k_space_sparse = np.zeros((nCol,nCol,nLineshs))

    tau = (1.0 + np.sqrt(5.0)) / 2.0
    golden_angle = np.pi / (tau + N - 1)
    steadystate_rays = nLin * (timeframe-1)
    offSet = golden_angle * (steadystate_rays)

    # Convert all the polar coordinates to cartesian coordinates
    for phase in range(0,nLineshs):

        l = 0
        
        for line in range(0,nLin):
            
            theta_radian = line*golden_angle + offSet
            theta_radian = np.mod(theta_radian, np.pi)
            
            k = 0

            for r in range(inPhases(-nCol/2), inPhases(nCol/2)):

                [x, y] = pol2cart(theta_radian, r);
                X[k,l] = x
                Y[k,l] = y
                k += 1

            l += 1
        
        grid_X[:,:,phase] = X
        grid_Y[:,:,phase] = Y


    # Apply gridding for each phase
    for phase in range(0,nLineshs):
    
        img = sparse_radial_data[:,:,phase]
        
        values = np.flipud(img)
        values = values.flatten()
        
        X = grid_X[:,:,phase].flatten()
        Y = grid_Y[:,:,phase].flatten()
        
        grid_x = np.round(X)
        grid_y = np.round(Y)

        poinPhasess = np.array((X,Y)).T
        
        ZI = griddata(poinPhasess, values, (grid_x, grid_y), method='nearest')
        
        KI = np.zeros(ZI.shape)
        ii = np.where(ZI>0)
        jj = np.where(ZI<=0)
        
        KI[ii] = ZI[ii]
        KI[jj] = ZI[jj]
        
        cart_k_space = KI
        
        cart_k_space_sparse = np.zeros((nCol,nCol))
        mask_k_space_sparse = np.zeros((nCol,nCol))
        
        temp_X = grid_x + nCol/2
        temp_Y = grid_y + nCol/2
        
        n_temp_X[:,:,phase] = temp_X.reshape((288,10))
        n_temp_Y[:,:,phase] = temp_Y.reshape((288,10))
        
        for i in range(0,nCol):
            for j in range(0,nLin):
                try:
                    cart_k_space_sparse[temp_Y[i,j], temp_X[i,j]] = cart_k_space[i,j]
                    mask_k_space_sparse[temp_Y[i,j], temp_X[i,j]] = 1
                except:
                    conPhasesinue
        
        g_cart_k_space_sparse[:,:,phase] = cart_k_space_sparse
        g_mask_k_space_sparse[:,:,phase] = mask_k_space_sparse

    return g_cart_k_space_sparse, g_mask_k_space_sparse

def main():
    
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'Raw', '2020_08_20_AALEM', 'meas_MID00524_FID122203_Radial_real_time_Cine_SAX.dat')

    twixObj = mapvbvd.mapVBVD(filename)

    print(twixObj)

    twixObj[1].image.flagRemoveOS = False # Preserve oversampling
    twixObj[1].image.squeeze = True       # Remove singleton dimensions

    # Extract the k-Space data
    kSpace = twixObj[1].image['']

    print('Shape of K-Space Data', kSpace.shape)

    kSpace = np.transpose(kSpace, (0,2,4,3,1))

    print('Shape of K-Space Data After Permutation', kSpace.shape)

    nSlices = kSpace.shape[3]
    nSamples = 4
    nCoils = 8
    frames_keep = 1

    for slice in range(0,nSlices):

      unstreaked_k_space = coil_unstreaking(kSpace[:,:,:,slice,:], nSamples, nCoils, frames_keep)

    exit()

    # Separate the inPhaseso complex data
    complexData = kSpace[:,0,0,0,0]

    phase = kSpace[0,0,0,:,0]

    for i in range(0,len(phase)):
        print(abs(phase[i]))

    print(phase)

    plt.scatter(kSpace[:,0,0,0,0],kSpace[:,0,:,0,0],marker='o',c='b',s=5)
    # plt.imshow(kSpace[100,0,5,:,:], cmap='plasma', vmin=0, vmax=2)
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.title("T1 maps")
    plt.savefig("mygraph.png")

    exit()

    # Organize data
    kSpace = np.transpose(kSpace, axes=[0, 2, 4, 3, 1])

    # Loop through cycles
    start_slice = 0
    end_slice = kSpace.shape[3]

    for slice_counPhaseser in range(start_slice, end_slice):
        coil = kSpace[:,:,:,slice_counPhaseser,:]

if __name__ == "__main__":
    main()