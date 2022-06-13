<img src='vids/video_1_lowres.gif' align="right" width=440>

<br><br><br><br><br><br>

# DRAPR
[Abstract](https://github.com/HMS-CardiacMR/RealTimeCine/tree/main/Abstract)

**DRAPR: Deep-Learning Radial Acceleration with Parallel Reconstruction**  

# Table of Contents 
- [OffLineIntegration](#Getting-Started)
- [InLineIntegration](#InLineIntegration)
- [Publications](#Publications)

## OffLineIntegration

Our code may be used to reconstruct raw k-space data offline. A Jupyter [notebook](https://github.com/HMS-CardiacMR/RealTimeCine/blob/main/notebooks/tutorial_basic.ipynb) tutorial is provided to demonstrate how to use our code. 


## InLineIntegration

<img src="vids/figure_2.png" width="800">

Multi-coil raw radial k-space data acquired from the scanner is processed in the Image Reconstruction Environment (ICE) on the vendor reconstruction computer. Using the International Society for Magnetic Resonance in Medicine Raw Data (ISMRMRD) format, the collected data is transferred to the [Framework for Image Reconstruction](https://github.com/kspaceKelvin/python-ismrmrd-server) (FIRE) server using a FireEmitter functor. The FIRE server is located in the vendor reconstruction computer. The data is then transferred from the FIRE server to an external server via a connecting Secure Shell Protocol (SSH) tunnel. In the external sever, a Docker containing all Python dependencies such as PyTorch is used to process the raw k-space data in a single 32 GB Graphics Processing Unit (GPU). First, a non-uniform fast Fourier transform (NUFFT) is used to grid and reconstruct undersampled multi-coil radial k-space data. GPU parallelization is done in PyTorch by treating frames and coils as batch and channel dimensions. This approach enables application of the NUFFT at 10 ms per frame. Coil sensitivity and combination is subsequently performed in PyTorch at negligible computational cost. These coil-combined images are send to the U-Net for de-aliasing, which requires 6.6 ms per frame. The total processing time of 16.6 ms is about half the 37.7 ms temporal resolution of collected frames. Images are then returned to the FIRE server via the same SSH tunnel, and to ICE using a FireInjector functor. Finally, the reconstructed de-aliased images are finalized into DICOM format and returned to the scanner computer console for immediate display.

#### Launching FIRE Server 

- Once the necessary libraries have been installed the FIRE server can be launched:
```bash
python main.py
```

You will then see the following message appear in your console:

```bash
2021-07-26 18:22:16,465 - Starting server and listening for data at 0.0.0.0:9002
```

This indicates the FIRE is running and will process any data sent to it via port 9002.

## Publications

If you use DRAPR or some part of the code, please cite:
- **An Inline Deep Learning Based Free-Breathing ECG-Free Cine for Exercise CMR.** [Manuel A. Morales](https://cardiacmr.hms.harvard.edu/people/manuel-morales-phd), [Salah Assana](https://cardiacmr.hms.harvard.edu/people/salah-assana), [Xiaoying Cai](https://cardiacmr.hms.harvard.edu/people/xiaoying-cai-phd), [Kelvin Chow](https://marketing.webassets.siemens-healthineers.com/1800000007010698/f017dc5c4ecd/Siemens-Healthineers-Meet_Healthineers_Kelvin_Chow_1800000007010698.pdf), [Hassan Haji-valizadeh](https://cardiacmr.hms.harvard.edu/people/hassan-haji-valizadeh-phd), [Eiryu Sai](https://cardiacmr.hms.harvard.edu/people/eiryu-sai-md-phd), [Connie Tsao](https://cardiacmr.hms.harvard.edu/people/connie-tsao), [Jason Matos](https://cardiacmr.hms.harvard.edu/people/jason-matos-md), [Jennifer Rodriguez](https://cardiacmr.hms.harvard.edu/people/jennifer-rodriguez), [Sophie Berg](https://cardiacmr.hms.harvard.edu/people/sophie-berg), [Neal Whitehead](https://cardiacmr.hms.harvard.edu/people/neal-whitehead-rn), [Patrick Pierce](https://cardiacmr.hms.harvard.edu/people/patrick-pierce), [Beth Goddu](https://cardiacmr.hms.harvard.edu/people/beth-goddu), [Warren J. Manning](https://cardiacmr.hms.harvard.edu/people/warren-j-manning), [Reza Nezafat](https://cardiacmr.hms.harvard.edu/people/reza-nezafat). Under revision.

- **Radial perfusion cardiac magnetic resonance imaging using deep learning image reconstruction.** [Salah Assana](https://cardiacmr.hms.harvard.edu/people/salah-assana), [Manuel A. Morales](https://cardiacmr.hms.harvard.edu/people/manuel-morales-phd), [Kei Nakata](https://cardiacmr.hms.harvard.edu/people/kei-nakata-md-phd), [Eiryu Sai](https://cardiacmr.hms.harvard.edu/people/eiryu-sai-md-phd), [Amine Amyar](https://cardiacmr.hms.harvard.edu/people/amine-amyar-phd), [Hassan Haji-valizadeh](https://cardiacmr.hms.harvard.edu/people/hassan-haji-valizadeh-phd), and [Reza Nezafat](https://cardiacmr.hms.harvard.edu/people/reza-nezafat). Poster [Presentation](https://submissions.mirasmart.com/ISMRM2022/Itinerary/ConferenceMatrixEventDetail.aspx?ses=G-169) at ISMRM 2022. *Inline implementation for perfusion coming soon!*

