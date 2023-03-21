# Inspecting Neural Style Transfer and Playaround :carousel_horse:

In this repository I have implemented original Neural Style Transfer paper "[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)" and inspected how result of transfer of content and style image changes by changing weight constants, learning rates, optimizers etc.

# Contents

1. [Introduction](#Introduction)
2. [Reconstruct](#Reconstruct)
   1. [Noise](##Noise)
   2. [Content](##Content)
   3. [Style](##Style)
   4. [Further-Studies](##Further-Studies)
3. [Visualization](#Visualization)
   1. [Style](##StyleV)
   2. [Content](##ContentV)
   3. [Both](##Both)

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

Early layers have composed style over canvas relatively well than higher layers but lost the semantics of content in terminal layers. Mid level layers have preserved the content while focusing less on style composition. 

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

Again early layers composed style over canvas relatively well than higher layers but while moving towards higher layers canvas is losing content representation maybe due to over composition of style. Last layer has again lost semantics to quite some extent. 

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

In first two rows the only change is in use of optimizer, and clearly both the optimizer produces comparitively similar canvas except in the last layer. Also `Adam` needs more iterations to produce semantically similar canvas to that of what `LBFGS` produce but at the same time former is quite fast to compute since its first order method and doesn't compute curvature of parameter space like latter. 

So we used `Adam` once again on different set of content and style image(last row) to generate canvas and found that last layer in all the cases loses some content information and style over composes on canvas. First two layers are giving comparitively better results all the time. 

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

Composing content representation over style canvas doesn't seem like a great idea. Last layers over composed the style with some noise, `content_layer: 2` smoothen out the background highlighting content. 

## Further Studies

From above experiments we can infer that in `content_layer: 4` canvas has lost semantics to some extent due to either *over composing of style* or *under-representation of content representation*. We can infer that out in [Visualization](##Visualization) by looking at what each layer is contributing in generating canvas.  The same can be said for `content_layer: 3` but with relatively less prominence than the former. 

In `content_layer: 0` we can see that style is well composed over canvas while also preserving the content representation, same can be said for `content_layer: 1` but with less prominence. So for further experiment lets' use `content_layer: 0` and `Adam` for fast computation. Currently we have seen all the canvases generated by `conv` layers, lets experiment with `relu` now.

| Content_Layers |                              0                               |                              1                               |                              2                               |                              3                               |                              4                               |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    **conv**    | ![0out0anim](https://user-images.githubusercontent.com/41532536/226406989-4726c0f8-2af7-4afe-82b5-3d6679bb6118.gif) | ![0out1anim](https://user-images.githubusercontent.com/41532536/226406897-0a35f296-d6e1-4b24-9e09-4dcf6b5e07f9.gif) | ![0out2anim](https://user-images.githubusercontent.com/41532536/226406816-c4912a9d-551e-43d4-9660-3124a47ca643.gif) | ![0out3anim](https://user-images.githubusercontent.com/41532536/226406738-d3a75d4a-3464-491a-934e-d753be6a71cd.gif) | ![0out4anim](https://user-images.githubusercontent.com/41532536/226406638-fb0a1c33-d49f-4fa2-9367-a2af6c0923fa.gif) |
|    **relu**    | ![0out0anim](https://user-images.githubusercontent.com/41532536/226403113-64d65811-7855-4fa0-b513-44adcfc22f01.gif) | ![0out1anim](https://user-images.githubusercontent.com/41532536/226403003-88e9f53c-73a4-4276-a0eb-c3ae19231bb8.gif) | ![0out2anim](https://user-images.githubusercontent.com/41532536/226402893-d9475902-e24c-4df2-ab68-af3b7e961988.gif) | ![0out3anim](https://user-images.githubusercontent.com/41532536/226402798-f80a739a-7e23-4ae3-87ee-c1b5fca3353e.gif) | ![0out4anim](https://user-images.githubusercontent.com/41532536/226402685-de4f134a-f95f-40ad-88f3-eb4835eb51c0.gif) |

looking at all the canvases from `conv` and `relu` we can infer that both the layers don't output very different canvases, and its safe to use either of layers for reconstruction.

# Visualization

Until now we have reconstructed canvases using all the style layers and any one content layer, but in this section we will **visualize** the individual and grouped contribution of style and content layers. We have 3 ways to do so, either only visualizing content layer(s), or visualizing style layers(s) or both layers.

shell command to visualize is

```bash
python3 main.py --visualize "content" --content_layers 1 2 --iterations 1500 --fps 30 --sav_freq 5
```

## ContentV

when `--visualize "content"` then we can only visualize the content representation of any layer or by grouping some layers.

| Content_Layers |                              0                               |                              1                               |                              2                               |                              3                               |                              4                               |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   **Canvas**   | ![0out0anim](https://user-images.githubusercontent.com/41532536/226537877-c7fe467f-2baf-43fa-9e9a-8a63a221176a.gif) | ![0out1anim](https://user-images.githubusercontent.com/41532536/226537868-51fa9f54-c503-4b0a-b799-5eb30c32386e.gif) | ![0out2anim](https://user-images.githubusercontent.com/41532536/226537858-11e649e9-d82c-4dad-beb3-e57f3d568132.gif) | ![0out3anim](https://user-images.githubusercontent.com/41532536/226537854-e22f57c9-33a0-46f1-9cd6-10d6e3ed285a.gif) | ![0out4anim](https://user-images.githubusercontent.com/41532536/226537841-99f23b78-e708-4648-95e4-42cc77f6dd90.gif) |

Latter layers are capturing textures of the content image while not giving much weightage to color and low level feature details. Although `content_layer: 4` canvas seems to have under-representated the content representation maybe due to insufficient number of gradients flowing back to canvas for update

Earlier layers captured the shape and somewhat texture really well. 

What if we arbitralily choose some content layers and find the output of their resultant on canvas, lets check

```bash
python3 main.py --visualize "content" --content_layers 1 3 4 ---iterations 700 --fps 2 --sav_freq 5
```

| Content_Layers | 1 3 4 | 0 2 4 |
| -------------- | ----- | ----- |
| **Canvas**     |   ![0out0anim](https://user-images.githubusercontent.com/41532536/226545739-189500ea-ca96-435d-a818-a21d9d799644.gif)    |   ![0out0anim](https://user-images.githubusercontent.com/41532536/226546315-5c6bf1c6-857c-4a27-a6c8-a8fe3d4e3363.gif)   |

## StyleV

when `--visualize "style"` then we can only visualize the style representation of any layer or by grouping some layers.

| Style_Layers | 0    | 1    | 2    | 3    | 4    |
| ------------ | ---- | ---- | ---- | ---- | ---- |
|   **Adam**   |   ![0out0anim](https://user-images.githubusercontent.com/41532536/226550689-225f86c1-476b-4007-99c9-3916e832e22f.gif)   |  ![0out1anim](https://user-images.githubusercontent.com/41532536/226550645-34662c8f-bec9-4837-9217-fea647a6932e.gif)    |  ![0out2anim](https://user-images.githubusercontent.com/41532536/226550609-8e39a25a-c535-4619-ab12-5689e581bacd.gif)    |   ![0out3anim](https://user-images.githubusercontent.com/41532536/226550525-f3de3c34-5195-494e-a8f7-ca345ffac49c.gif)   |   ![0out4anim](https://user-images.githubusercontent.com/41532536/226550453-a2f40922-45b0-4074-9c33-bf1d6231931e.gif)   |

Style layers when visualized individualy seems to have not been contributing any significant style to canvas, infact while moving towards higher layers we see patterns of noise.

What if we arbitralily choose some content layers and find the output of their resultant on canvas, lets check

```bash
python3 main.py --visualize "style" --content_layers 1 3 4 ---iterations 2000 --fps 25 --sav_freq 8 --optimizer "Adam"
```

| Style_Layers | 0 1 4 | 1 2 3 | 0 1   |
| ------------ | ----- | ----- | ----- |
|   **Adam**   |   ![0out0anim](https://user-images.githubusercontent.com/41532536/226555881-84002b04-78d9-43c1-afb7-0a9df0f33755.gif)    |   ![0out1anim](https://user-images.githubusercontent.com/41532536/226555948-8776095b-a2bc-4092-8e15-9407ca5cf1f9.gif)    |   ![0out2anim](https://user-images.githubusercontent.com/41532536/226556012-57a80c37-a5b4-46f7-b1d8-43de5e804e3d.gif)    |
|   **LBFGS**  |   ![0out0anim](https://user-images.githubusercontent.com/41532536/226560740-6a4362aa-0447-4bfc-9543-7c291b1595ea.gif)    |    ![0out1anim](https://user-images.githubusercontent.com/41532536/226560684-44758ca3-6554-4df9-a96b-ee15e0f86cdf.gif)   |   ![0out2anim](https://user-images.githubusercontent.com/41532536/226560639-a8c42f87-2fb7-41ea-bb5a-7d03b2edb0ca.gif)    |

<center>
    <table>
        <tr >
            <td>
                <center>
            	<img src="https://user-images.githubusercontent.com/41532536/226561065-beff04d0-0057-4199-989a-4ef3305cfd66.gif"/>
                </center>
            </td>
        </tr>
        <tr>
            <td>
                <center>
            		canvas output when all the style layers were used
                </center>
            </td>
        </tr>
    </table>
</center>

When visualized grouped contribution of layers we can see some style over canvas very clearly. `LBGFS` shows style in every canvas even when `Adam` failed to in `style_layers: 1 2 3`. On further looking into the matter we found that Adam too atleast 4000 iterations to learn the representations and output visually appealing style in comparision to others. The reason behind it can be that higher layers don't focus more on colors but on texture and `Adam` find it hard to extract the color features information than `LBFGS`. 

In the last we can visualize what all the style layers are contributing to the canvas, it looks more similar to style image itself.

## Both

For fun we will use all the style and content layers to generate the canvas, although this configuration worked for the below image but no for many.

<p align="center">
    <img src="https://user-images.githubusercontent.com/41532536/226572788-f0ad7ccf-4b80-42db-8894-3afed210e1d3.gif"/>
</p>

Original image of lion was grey.

You can play with with other hyperparameters to generate canvases and enhance your understanding of Neural Style Transfer
