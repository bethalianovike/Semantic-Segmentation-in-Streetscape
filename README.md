# Semantic Segmentation in Streetscape
Data Science Project 2020

## Requirements
* PyTorch 1.x or 0.41

## Installation
```
pip install -r requirements.txt
```

## Training on Cityscapes Dataset
1. Download dataset from 
original image: [here](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
file annotation image: [here](https://www.cityscapes-dataset.com/file-handling/?packageID=1) 
to inputs/ and unzip.
File structure:
```
inputs
└── cityscapes
    ├── images
    |   ├── 0a7e06.png
    │   ├── 0aab0a.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 0a7e06.png
        |   ├── ...
        |
        ├── 1
        |   ├── 0a7e06.png
        |   ├── ...
        |
        ├── ...
        | 
        └── 27
            ├── 0a7e06.png
            ├── ...
```
2. Preprocess
3. Train model
* with Deep Supervision
```
python train.py --dataset cityscapes --arch NestedUNet --num_classes 29 --deep_supervision True --epochs 200
``` 
* without Deep Supervision
```
python train.py --dataset cityscapes --arch NestedUNet --num_classes 29 --deep_supervision False --epochs 200
```
4. Evaluate and Visualize
* with Deep Supervision
```
python val.py --name cityscapes_NestedUNet_wDS
```
* without Deep Supervision
```
python val.py --name cityscapes_NestedUNet_woDS
```

## Results
### Cityscape Dataset (Image size: 512x256)
* **Loss Function: BCEDiceLoss**

| Model                           | IoU        | Dice Coeff |
| ------------------------------- | ---------- | ---------- |
| UNet++ without Deep Supervision | **0.7934** | **0.8785** |
| UNet++ with Deep Supervision    | 0.7767     | 0.8626     |
| UNet                            | 0.7920     | 0.8776     |

### Cityscape Dataset

| Model                           | IoU        | Dice Coeff |
| ------------------------------- | ---------- | ---------- |
| UNet++ without Deep Supervision | 0.7934     | **0.8785** |
| PSPNet                          | 0.8628     | 0.6771     |
| HRNet-ORC                       | **0.8886** | 0.7405     |
| UNet                            | 0.7920     | 0.8776     |
