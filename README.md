# Multi-Dilated-Fourier-Autoformer
This repository contains the code for the Multi-Scale Dilated Autoformer for UAV Energy Consumption Forecasting”. (will be update after the paper published)

## Requirements
We run the project on:
- PyTorch==2.7.0+cu118 on CUDA 11.8 
- Python==3.10.18
- Torchvision==0.22.0+cu118

#### **Step 1.** Clone and install requirements
* Clone this repo:
```
git clone https://github.com/zalzakarima/Multi-Dilated-Fourier-Autoformer.git
cd Multi-Dilated-Fourier-Autoformer
```
* Create a new virtual environment using Conda or virtualenv. 
```
conda create --name <envname> python=3.10
```
* Activate the environment and install the requirements:
```
conda activate <envname>
pip install -r requirements.txt
```

#### **Step 2.** Training
```
python train.py
```

#### **Step 3.** Evaluating
```
python evaluate.py
```

# Contact
For questions about our paper or code, please contact zalzakarima@kookmin.ac.kr.
Hi