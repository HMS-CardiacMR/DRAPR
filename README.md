# Reconstruction of Golden Angle Radial MRI 

<img src='vids/video_1_lowres.gif' align="right" width=440>

<br><br><br><br>

**Purpose**: To develop and evaluate a free-breathing and ECG-free real-time cine with deep learning (DL)-based radial acceleration for exercise CMR.

**Method**: We implemented a 3D (2D+time) convolutional neural network to suppress streaking artifacts from undersampled radial cine images. We trained the network using synthetic real-time radial cine images simulated using ECG-gated segmented Cartesian k-space data, which was acquired from 503 patients during breath-hold and at rest. Further, we implemented a prototype real-time radial sequence with acceleration rate = 12 on a 3T scanner, and used it to collect cine images with inline DL reconstruction whose total reconstruction time was 16.6 ms per frame. We evaluated the performance of the proposed approach by initially recruiting 9 healthy subjects in whom only rest images were collected. Subsequently, we recruited 14 subjects who participated in an exercise CMR imaging protocol in which both rest and post-exercise images were collected, including 8 patients with suspected coronary artery disease. Exercise was done using a CMR-compatible supine cycle ergometer positioned on the MR table.

**Results**: the DL model substantially suppressed streaking artifacts in free-breathing ECG-free real-time cine images acquired at rest and during post-exercise stress. Three readers evaluated residual artifact level in the entire field-of-view on a 3-point Likert scale (1-severe, 2-moderate, 3-minimal). In real-time images at rest, 89.4% of scores were moderate to minimal. The mean score was 2.3 ± 0.7, representing a significantly (P<0.001) higher amount of artifacts compared to ECG-gated (2.9 ± 0.3). In real-time images during post-exercise stress, 84.6% of scores were moderate to minimal, and the mean artifact level score was 2.1 ± 0.6. Comparison of left-ventricular measures derived from ECG-gated segmented and real-time cine images at rest showed differences in end-diastolic volume (-3.0 mL [-17.8 11.7], P=0.540) and mass (-0.6 g [-13.4 12.2], P=0.880) that were not significantly different from zero. Differences in measures of end-systolic volume (-7.0 mL [-15.3 1.3], P=020) and ejection fraction (5.0% [-1.0 11.1], P=0.037) were significant.

**Conclusions**: Our feasibility study demonstrated the potential of inline real-time cine with DL-based radial acceleration for Ex-CMR.
