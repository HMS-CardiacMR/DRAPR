# DRAPR: Deep-Learning Radial Acceleration with Parallel Reconstruction

*We implemented a 3D (2D+time) convolutional neural network to suppress streaking artifacts from undersampled radial cine images. We trained the network using synthetic real-time radial cine images simulated using ECG-gated segmented Cartesian k-space data, which was acquired from 503 patients during breath-hold and at rest. Further, we implemented a prototype real-time radial sequence with acceleration rate = 12 on a 3T scanner, and used it to collect cine images with inline DL reconstruction whose total reconstruction time was 16.6 ms per frame. We evaluated the performance of the proposed approach by initially recruiting 9 healthy subjects in whom only rest images were collected. Subsequently, we recruited 14 subjects who participated in an exercise CMR imaging protocol in which both rest and post-exercise images were collected, including 8 patients with suspected coronary artery disease. Exercise was done using a CMR-compatible supine cycle ergometer positioned on the MR table.*

# Abstract

**Purpose**: To develop and evaluate a free-breathing and ECG-free real-time cine with deep learning (DL)-based radial acceleration for exercise CMR.

**Method**: We implemented a 3D (2D+time) convolutional neural network to suppress streaking artifacts from undersampled radial cine images. We trained the network using synthetic real-time radial cine images simulated using ECG-gated segmented Cartesian k-space data, which was acquired from 503 patients during breath-hold and at rest. Further, we implemented a prototype real-time radial sequence with acceleration rate = 12 on a 3T scanner, and used it to collect cine images with inline DL reconstruction whose total reconstruction time was 16.6 ms per frame. We evaluated the performance of the proposed approach by initially recruiting 9 healthy subjects in whom only rest images were collected. Subsequently, we recruited 14 subjects who participated in an exercise CMR imaging protocol in which both rest and post-exercise images were collected, including 8 patients with suspected coronary artery disease. Exercise was done using a CMR-compatible supine cycle ergometer positioned on the MR table.

**Results**: the DL model substantially suppressed streaking artifacts in free-breathing ECG-free real-time cine images acquired at rest and during post-exercise stress. Three readers evaluated residual artifact level in the entire field-of-view on a 3-point Likert scale (1-severe, 2-moderate, 3-minimal). In real-time images at rest, 89.4% of scores were moderate to minimal. The mean score was 2.3 ± 0.7, representing a significantly (P<0.001) higher amount of artifacts compared to ECG-gated (2.9 ± 0.3). In real-time images during post-exercise stress, 84.6% of scores were moderate to minimal, and the mean artifact level score was 2.1 ± 0.6. Comparison of left-ventricular measures derived from ECG-gated segmented and real-time cine images at rest showed differences in end-diastolic volume (-3.0 mL [-17.8 11.7], P=0.540) and mass (-0.6 g [-13.4 12.2], P=0.880) that were not significantly different from zero. Differences in measures of end-systolic volume (-7.0 mL [-15.3 1.3], P=020) and ejection fraction (5.0% [-1.0 11.1], P=0.037) were significant.

**Conclusions**: Our feasibility study demonstrated the potential of inline real-time cine with DL-based radial acceleration for Ex-CMR.

## Off-Line Implenetation

Our code may be used to reconstruct raw k-space data offline. A Jupyter [notebook](https://github.com/HMS-CardiacMR/RealTimeCine/blob/main/notebooks/tutorial_basic.ipynb) tutorial is provided to demonstrate how to use our code. 

## In-Line Implenetation

<img src="vids/figure_2.png" width="100%">

Multi-coil raw radial k-space data acquired from the scanner is processed in the Image Reconstruction Environment (ICE) on the vendor reconstruction computer. Using the International Society for Magnetic Resonance in Medicine Raw Data (ISMRMRD) format, the collected data is transferred to the [Framework for Image Reconstruction](https://github.com/kspaceKelvin/python-ismrmrd-server) (FIRE) server using a FireEmitter functor. The FIRE server is located in the vendor reconstruction computer. The data is then transferred from the FIRE server to an external server via a connecting Secure Shell Protocol (SSH) tunnel. In the external sever, a Docker containing all Python dependencies such as PyTorch is used to process the raw k-space data in a single 32 GB Graphics Processing Unit (GPU). First, a non-uniform fast Fourier transform (NUFFT) is used to grid and reconstruct undersampled multi-coil radial k-space data. GPU parallelization is done in PyTorch by treating frames and coils as batch and channel dimensions. This approach enables application of the NUFFT at 10 ms per frame. Coil sensitivity and combination is subsequently performed in PyTorch at negligible computational cost. These coil-combined images are send to the U-Net for de-aliasing, which requires 6.6 ms per frame. The total processing time of 16.6 ms is about half the 37.7 ms temporal resolution of collected frames. Images are then returned to the FIRE server via the same SSH tunnel, and to ICE using a FireInjector functor. Finally, the reconstructed de-aliased images are finalized into DICOM format and returned to the scanner computer console for immediate display.

### Launching FIRE Server 

It is recommended to first create a virtual enviornment prior to running the code.

Create Python venv with Python:

     python -m venv drapr-venv

Activate Virtual Enviornment:

    source drapr-venv/bin/activate

Install dependinces:

    pip install -r requirements.txt

Once the necessary libraries have been installed the FIRE server can be launched:

    python main.py

You will then see the following message appear in your console:

    2021-07-26 18:22:16,465 - Starting server and listening for data at 0.0.0.0:9002

This indicates the FIRE is running and will process any data sent to it via port 9002.

## Comparison to GRASP

<img src='vids/video_1_lowres.gif' align="left" width=100%>

## Application to Exercise CMR

<img src='vids/video_3_lowres.gif' align="left" width=100%>

## Application to Perfusion Imaging

[![Radial Perfusion Cardiac Magnetic Resonance Imaging Using Deep Learning Image Reconstruction](https://img.youtube.com/vi/2r85g0-Mwis/maxresdefault.jpg)](https://www.youtube.com/watch?v=2r85g0-Mwis "Click to play video")

## Publications

If you use DRAPR or some part of the code, please cite:
- **An Inline Deep Learning Based Free-Breathing ECG-Free Cine for Exercise CMR.** [Manuel A. Morales](https://cardiacmr.hms.harvard.edu/people/manuel-morales-phd), [Salah Assana](https://salahassana.com/), [Xiaoying Cai](https://cardiacmr.hms.harvard.edu/people/xiaoying-cai-phd), [Kelvin Chow](https://marketing.webassets.siemens-healthineers.com/1800000007010698/f017dc5c4ecd/Siemens-Healthineers-Meet_Healthineers_Kelvin_Chow_1800000007010698.pdf), [Hassan Haji-valizadeh](https://cardiacmr.hms.harvard.edu/people/hassan-haji-valizadeh-phd), [Eiryu Sai](https://cardiacmr.hms.harvard.edu/people/eiryu-sai-md-phd), [Connie Tsao](https://cardiacmr.hms.harvard.edu/people/connie-tsao), [Jason Matos](https://cardiacmr.hms.harvard.edu/people/jason-matos-md), [Jennifer Rodriguez](https://cardiacmr.hms.harvard.edu/people/jennifer-rodriguez), [Sophie Berg](https://cardiacmr.hms.harvard.edu/people/sophie-berg), [Neal Whitehead](https://cardiacmr.hms.harvard.edu/people/neal-whitehead-rn), [Patrick Pierce](https://cardiacmr.hms.harvard.edu/people/patrick-pierce), [Beth Goddu](https://cardiacmr.hms.harvard.edu/people/beth-goddu), [Warren J. Manning](https://cardiacmr.hms.harvard.edu/people/warren-j-manning), [Reza Nezafat](https://cardiacmr.hms.harvard.edu/people/reza-nezafat). Under revision.

- **Radial perfusion cardiac magnetic resonance imaging using deep learning image reconstruction.** [Salah Assana](https://salahassana.com/), [Manuel A. Morales](https://cardiacmr.hms.harvard.edu/people/manuel-morales-phd), [Kei Nakata](https://cardiacmr.hms.harvard.edu/people/kei-nakata-md-phd), [Eiryu Sai](https://cardiacmr.hms.harvard.edu/people/eiryu-sai-md-phd), [Amine Amyar](https://cardiacmr.hms.harvard.edu/people/amine-amyar-phd), [Hassan Haji-valizadeh](https://cardiacmr.hms.harvard.edu/people/hassan-haji-valizadeh-phd), and [Reza Nezafat](https://cardiacmr.hms.harvard.edu/people/reza-nezafat). Poster [Presentation](https://submissions.mirasmart.com/ISMRM2022/Itinerary/ConferenceMatrixEventDetail.aspx?ses=G-169) at ISMRM 2022. *Inline implementation for perfusion coming soon!*

