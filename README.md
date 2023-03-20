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

bash command e.g,

```bash
python3 main.py --reconstruct --style_layers 0 1 2 3 4 --content_layers 1 --optimizer "Adam"
```

parameters we are using

```yaml
optimizer: "Adam" 
init_image: "noise"
```

|    Content_Layer     |                              0                               |                              1                               |                              2                               |                              3                               |                              4                               |
| :------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Generated Canvas** | ![0out(1)anim](https://user-images.githubusercontent.com/41532536/226229173-eab555a5-cabd-4fb2-9cac-252ed482e88c.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226230389-dfb97a0f-4b3f-4ca3-9068-f8af45f98be0.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226230917-e317265a-6a3b-47a3-9c2a-283b2ca36b12.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226231131-f910b4ca-57fd-4c2c-a571-2d603752e8e3.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226233226-e4eb9446-0af0-43d2-804f-60f56bb10f1c.gif) |

on `A4000` GPU it took 33s to run with current configuration for one canvas generation

### Changing Optimizer

```bash
python3 main.py --reconstruct --style_layers 0 1 2 3 4 --content_layers 0 --iterations 2000
```

parameters we using

```yaml
optimizer: "LBFGS"
init_image: "noise"
```

|    Content_Layer     |                              0                               |                              1                               |                              2                               |                              3                               |                              4                               |
| :------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Generated Canvas** | ![0outanim](https://user-images.githubusercontent.com/41532536/226241745-a17e30a3-49f1-429f-8a48-b34f7cb4be62.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226241394-a003ccc4-40d3-40c6-9a43-6b88582929cc.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226240585-7320d8b9-f536-4c0e-83e3-8ad0030cd299.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226237980-83c8d273-739a-4361-84c8-5d786d12691c.gif) | ![0outanim](https://user-images.githubusercontent.com/41532536/226237203-f7984ea4-646e-48ad-b42f-6e6eaef8b609.gif) |

pn `A4000` GPU it took 120s to run with current configuration for one canvas generation

## Content


We can initialize the canvas with **content image** itself and then update the values to look similar to the content image having style composed on it. Using below line of code we initiate canvas with content image.

```python
generated_image = content_image.clone().requires_grad_(True)
```

lets' start with some experiments...:microscope:

### Changing Optimizer

bash command e.g,

```bash
python3 main.py --reconstruct --style_layers 0 1 2 3 4 --content_layers 1 --optimizer "Adam" --init_image "content"
```

| Content_Layers | 0    | 1 | 2 | 3 | 4 |
| :------------: | :--: | :--: | :--: | :--: | :--: |
| **Adam**       |   ![0outanim](https://user-images.githubusercontent.com/41532536/226280510-eb297ac3-8e43-44c2-9e00-305c24ac9d8b.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226281106-fd98c3c2-9768-4ec1-95be-0943598eeba2.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226281629-69d7d893-b822-4283-933a-c42ec1782088.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226282069-1b45fb12-fc58-4b9b-8cbe-5065beebc7f2.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226282647-625c87c4-7aca-45a5-badc-256bbc4b85ab.gif)   |
| **LBFGS**      |   ![0outanim](https://user-images.githubusercontent.com/41532536/226283096-7428c3f7-f4e5-4089-adfb-4c1ef5251aac.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226283411-a19e6bd3-3b0b-4caf-af3c-63ae78e15036.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226284077-70246242-e650-4fa7-99c1-8ac37ab3e58c.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226284517-b9611e75-ce19-4229-b315-8a1db9be3f0d.gif)   |  ![0outanim](https://user-images.githubusercontent.com/41532536/226285297-642c40af-a15b-4a64-aa35-1ad613499ec1.gif)    |
| **Adam**      |  ![0outanim](https://user-images.githubusercontent.com/41532536/226285822-c21edaf0-2e80-448e-acf9-2bd283e3eb58.gif)    |  ![0outanim](https://user-images.githubusercontent.com/41532536/226286320-4af4a5f0-dad8-4646-b64f-14dbe7cb682a.gif)    |  ![0outanim](https://user-images.githubusercontent.com/41532536/226286564-c357f8dd-2008-4ad2-8543-8567ed703a7f.gif)    |   ![0outanim](https://user-images.githubusercontent.com/41532536/226286843-48fce497-77c0-4afe-bb29-af954a97dd9f.gif)   |   ![0outanim](https://user-images.githubusercontent.com/41532536/226287009-4782cadc-39c0-40db-8068-e50b67ce7b77.gif)   |


## Style

We can initialize the canvas with **style image** itself and then update the values to look similar to the content image having style composed on it. Using below line of code we initiate canvas with content image.

```python
generated_image = style_image.clone().requires_grad_(True)
```

lets' start with some experiments... :microscope:

### Changing Optimizer

```bash
python3 main.py --reconstruct --style_layers 0 1 2 3 4 --content_layers 1 --optimizer "Adam" --init_image "style"
```

| Content_Layers |  0   |  1   |  2   |  3   |  4   |
| :------------: | :--: | :--: | :--: | :--: | :--: |
|    **Adam**    |   ![0out0anim](https://user-images.githubusercontent.com/41532536/226337250-8b09bec5-bf42-4930-951b-337c0aeab814.gif)   |   ![0out1anim](https://user-images.githubusercontent.com/41532536/226337180-c499d371-92df-4676-bcb6-e754b560283e.gif)   |   ![0out2anim](https://user-images.githubusercontent.com/41532536/226337029-f98f031a-b7c1-4fbe-9f8d-eeb54d2d089e.gif)   |   ![0out3anim](https://user-images.githubusercontent.com/41532536/226336876-539604fc-907f-4e20-97ff-d1b1e1a4f560.gif)   |   ![0out4anim](https://user-images.githubusercontent.com/41532536/226336780-488389e4-5516-4f77-a11b-e23ae5a3ae60.gif)   |


# Visualization



