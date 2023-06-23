# Image classification - CIFAR add.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/), 
[PyTorch](https://pytorch.org/), 
[torchvision](https://github.com/pytorch/vision) 0.8, 
Uses [matplotlib](https://matplotlib.org/)  for ploting accuracy and losses.

## Info

 * we are training our model on CIFAR dataset. 
 * we are using a custom convolutional neural network (CNN) architectures. 
 * Implemented in pytorch 

## About

* we trained the following model using Batch norm(BN), Group norm(GN), Layer norm(LN) for comparision 
* we are trying to debug the notion of BN being superior in the context of cnns as each channel in a given layer tries to extract more or less the same feature. 
* model has all three models written. as we are using nn.Squential i had to make 3 classes for each. 
* all utility function related to models are in model.py and other utility functions can be found in utils.py. 
* also added incorrect_data function inspired from parrotletml git. 

## Results 

### Train accuracy 

* BN - 80.49%
* GN - 78.59%
* LN - 79.55%

### Test accuracy 

* BN - 75.77%
* GN - 74.18%
* LN - 74.98%

## Usage

```bash
git clone https://github.com/srikanthp1/S8.git
```
* utils.py has util functions
* model.py has models 
* run cell by cell to download, visualize data and train model


## Model details

```python
model = Net().to(device)
summary(model, input_size=(3, 32, 32))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 30, 30]             432
              ReLU-2           [-1, 16, 30, 30]               0
       BatchNorm2d-3           [-1, 16, 30, 30]              32
           Dropout-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 20, 28, 28]           2,880
              ReLU-6           [-1, 20, 28, 28]               0
       BatchNorm2d-7           [-1, 20, 28, 28]              40
           Dropout-8           [-1, 20, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]             320
        MaxPool2d-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 20, 14, 14]           2,880
             ReLU-12           [-1, 20, 14, 14]               0
      BatchNorm2d-13           [-1, 20, 14, 14]              40
          Dropout-14           [-1, 20, 14, 14]               0
           Conv2d-15           [-1, 24, 14, 14]           4,320
             ReLU-16           [-1, 24, 14, 14]               0
      BatchNorm2d-17           [-1, 24, 14, 14]              48
          Dropout-18           [-1, 24, 14, 14]               0
           Conv2d-19           [-1, 28, 14, 14]           6,048
             ReLU-20           [-1, 28, 14, 14]               0
      BatchNorm2d-21           [-1, 28, 14, 14]              56
          Dropout-22           [-1, 28, 14, 14]               0
           Conv2d-23           [-1, 16, 14, 14]             448
        MaxPool2d-24             [-1, 16, 7, 7]               0
           Conv2d-25             [-1, 20, 7, 7]           2,880
             ReLU-26             [-1, 20, 7, 7]               0
      BatchNorm2d-27             [-1, 20, 7, 7]              40
          Dropout-28             [-1, 20, 7, 7]               0
           Conv2d-29             [-1, 24, 7, 7]           4,320
             ReLU-30             [-1, 24, 7, 7]               0
      BatchNorm2d-31             [-1, 24, 7, 7]              48
          Dropout-32             [-1, 24, 7, 7]               0
           Conv2d-33             [-1, 28, 7, 7]           6,048
             ReLU-34             [-1, 28, 7, 7]               0
      BatchNorm2d-35             [-1, 28, 7, 7]              56
          Dropout-36             [-1, 28, 7, 7]               0
        AvgPool2d-37             [-1, 28, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             280
================================================================
Total params: 31,216
Trainable params: 31,216
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.61
Params size (MB): 0.12
Estimated Total Size (MB): 1.74
----------------------------------------------------------------

```

## Analysis 

* After training for 15 epochs BN gave the highest accuracy though the margin is less than 2% 

* BN converged to 60% quickly than others. after pretty much saturation started. so we can assume BN helps with convergence. 

* this may just have to do with this experiment but LN converged smootly compared to BN which has a bump at 8th epach. 

