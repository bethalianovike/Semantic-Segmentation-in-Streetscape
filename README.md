# Semantic-Segmentation-in-Streetscape
Data Science Project 2020

## Requirements
* PyTorch 1.x or 0.41

## Installation
`pip install -r requirements.txt`

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
`python train.py --dataset cityscapes --arch NestedUNet --num_classes 29 --deep_supervision True --epochs 200` -> with Deep Supervision
`python train.py --dataset cityscapes --arch NestedUNet --num_classes 29 --deep_supervision False --epochs 200` -> without Deep Supervision
4. Evaluate and Visualize
`python val_beth.py --name cityscapes_NestedUNet_wDS` -> with Deep Supervision
`python val_beth.py --name cityscapes_NestedUNet_woDS` -> without Deep Supervision
