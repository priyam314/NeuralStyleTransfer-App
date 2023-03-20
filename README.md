# Inspecting Neural Style Transfer and Playaround :carousel_horse:

In this repository I have implemented original Neural Style Transfer paper "[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)" and inspected how result of transfer of content and style image changes by changing weight constants, learning rates, optimizers etc.

# Contents

1. [Introduction](#Introduction)
2. [Reconstruct](#Reconstruct)
   1. [Noise](##Noise)
   2. [Content](##Content)
   3. [Style](##Style)
3. Visualization
   1. Style
   2. Content
4. 

# Introduction

Style Transfer is the task of composing style from one image which is *style image* over another image which is *content image*. Before doing style transfer using neural network the major limiting factor in this task was feature representation of content and style image for better composition. Lack of such representations thwarted the way to understand the semantics and separation between the two. With the success :heavy_check_mark: of VGG networks on ImageNet Challenge in Object Localization and Object Detection :mag: , researchers gave the style transfer a neural approach.



Authors used the feature representations from VGG network to learn high and low level features of both content and style images. Using these implicit information they kept minimizing the loss between content representation and generated image representation using **MSELoss** and between style representation and generated image representation using  **MSELoss of Gram Matrices**. Task of Neural Style Transfer unlike supervised learning doesn't have metric to compare performance of quality of image(s). We are not training model but updating the values of image itself in every iteration using gradient descent such that it match closely with content and style image.



I believe this brief overview of Neural Style Transfer is enough to get us started with experiments and notice some fascinating results.

*Note:* This is not a blog post on Neural Style Transfer. No exlpanation on the type of model, training etc is provided.

# Setting Parameters

For our experiments we will set the parameters to following value until explicitly written.

```yaml
iterations: 2500
fps: 30
size: 128
sav_freq: 10
alpha: 5.0
beta: 7000.0
gamma: 1.2
style_weights: [1e3/n**2 for n in [16.0,32.0,128.0,256.0,512.0]]
lr: 0.06
```

if path to content image and style images are not provided then default images will be used that lie inside `NeuraltyleTransfer-App/src/data`

For detailed understanding about these parameters go through `python3 main.py -h`

# Reconstruct

Neural Style Transfer is like painting an image over a canvas. This canvas is of same size to that of content image since content is static and only dynamic changes that need to be composed over this canvas is of style image. Though size is same to that of content image but there are 3 - 4 ways we can initialize this canvas with, and then using gradient descent :chart_with_downwards_trend: update the values of the canvas.

Following shell command can lead you to generate canvas by blending the style over content image. This is basic bash command for reconstruction of canvas, for more infomation about arguments you can go through `python3 main.py --help`

```bash
python3 main.py --reconstruct --content_layers <num> --style_layers 0 1 2 3 4
```

## Noise

We can initialize the canvas with **noise** and then update the values to look similar to the content image having style composed on it. Using below script we generate a noise canvas and set its `requires_grad = True`. This enables the grad function to update the values of the following canvas. 

```python
generated_image = torch.randn(content_image.size())
generated_image.to(device, torch.float)
generated_image.requires_grad = True
```

Lets start with some experiments... :microscope:

### Changing Content Layers

parameters we are using

```yaml
optimizer: "Adam"
init_image: "noise"
```

| Content_Layer    | 0           | 1    | 2    | 3    | 4    |
| :--------------- | ----------- | ---- | ---- | ---- | ---- |
| Generated Canvas | ![0out(1)anim](https://user-images.githubusercontent.com/41532536/226229173-eab555a5-cabd-4fb2-9cac-252ed482e88c.gif) |      |      |      |      |





## Content

We can initialize the canvas with **content image** itself and then update the values to look similar to the content image having style composed on it. Using below line of code we initiate canvas with content image.

```python
generated_image = content_image.clone().requires_grad_(True)
```



## Style

We can initialize the canvas with **style image** itself and then update the values to look similar to the content image having style composed on it. Using below line of code we initiate canvas with content image.

```python
generated_image = style_image.clone().requires_grad_(True)
```



# Visualization



