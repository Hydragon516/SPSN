# SPSN: Superpixel Prototype Sampling Network for RGB-D Salient Object Detection (ECCV 2022)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spsn-superpixel-prototype-sampling-network/rgb-d-salient-object-detection-on-sip)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-sip?p=spsn-superpixel-prototype-sampling-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spsn-superpixel-prototype-sampling-network/rgb-d-salient-object-detection-on-des)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-des?p=spsn-superpixel-prototype-sampling-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spsn-superpixel-prototype-sampling-network/rgb-d-salient-object-detection-on-nju2k)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-nju2k?p=spsn-superpixel-prototype-sampling-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spsn-superpixel-prototype-sampling-network/rgb-d-salient-object-detection-on-nlpr)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-nlpr?p=spsn-superpixel-prototype-sampling-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spsn-superpixel-prototype-sampling-network/rgb-d-salient-object-detection-on-stere)](https://paperswithcode.com/sota/rgb-d-salient-object-detection-on-stere?p=spsn-superpixel-prototype-sampling-network)

Authors: [Minhyeok Lee](https://github.com/Hydragon516), [Chaewon Park](https://github.com/codnjsqkr), [Suhwan Cho](https://github.com/suhwan-cho), Sangyoun Lee

This repository provides code for paper "SPSN: Superpixel Prototype Sampling Network for RGB-D Salient Object Detection" accepted by the ECCV 2022 conference.

Our paper can be found [[arXiv]](https://arxiv.org/abs/2207.07898)

<img align="center" src="./images/main.png" width="800px" />

## Prepared Datasets
Download the train and test dataset from [Google Drive](https://drive.google.com/file/d/17Ee2l1837HkHR8EGoR4u1Be3v_qliXj0/view?usp=sharing) and save it at your workspace.

## Requirements
For the superpixel algorithm we use [fast_slic](https://github.com/Algy/fast-slic). You can install it like this:
```
pip install fast_slic
```

## Training Model
1. First, clone this repository.
```
git clone https://github.com/Hydragon516/SPSN
```
2. Edit config.py. The data root path option and GPU index should be modified.
3. Train the model.
```
python3 train.py
```

## Evaluation
When training is complete, the prediction results for the test set are saved in the ./log folder. Two popular evaluation toolboxes are available. (Matlab version: https://github.com/DengPingFan/CODToolbox Python version: https://github.com/lartpang/PySODMetrics)

## Result
<img align="center" src="./images/result.png" width="600px" />

The prediction mask results for our proposed model can be found [here](https://drive.google.com/file/d/1QjgsNz7S21yNIbCsUW3zINmivxXr6vK0/view?usp=sharing).
