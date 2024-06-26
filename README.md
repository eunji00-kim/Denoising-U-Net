# Denoising-U-Net

* Implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* Original U-Net is a segmentation model, but this version works for **Denoising**.
* referred from https://github.com/SSinyu/RED-CNN

![image](https://github.com/eunji00-kim/Denoising-U-Net/assets/101125771/2fc1466b-4827-4e3e-9b66-1b9b29c1589e)


---------------------------------------

## Dataset

The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic  
<https://www.aapm.org/GrandChallenge/LowDoseCT/>

The ```data_path``` should look like:

```
data_path
├── L067
│   ├── quarter_3mm
│   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
│   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
│   │       └── ...
│   └── full_3mm
│           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
│           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
│           └── ...
├── L096
│   ├── quarter_3mm
│   │       └── ...
│   └── full_3mm
│           └── ...      
...
│
└── L506
    ├── quarter_3mm
    │       └── ...
    └── full_3mm
            └── ...
```

---------------------------------------

## Version Information

```
python==3.8
torch==1.13.1
torchvision==0.14.1
numpy==1.24.4
pydicom==2.4.4
matplotlib==3.7.5
```

---------------------------------------

## Combined U-Net with V-Net
* Taking an idea of [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) to improve performances
* The file name of the code is ```uunet.py```

![image](https://github.com/eunji00-kim/Denoising-U-Net/assets/101125771/0acac631-3ad4-4e7d-afbe-d47b4464fad3)

---------------------------------------

# Usage
* 1. Run ```dicom_to_numpy.py``` to dicom file to numpy.
  2. Run ```train.py```
  3. Run ```test.py```
