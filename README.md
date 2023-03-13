# Inspecting Neural Style Transfer and Playaround :carousel_horse:

In this repository I have implemented original Neural Style Transfer paper "[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)" and inspected how result of transfer of content and style image changes by changing weight constants, learning rates, optimizers etc.

# Contents

1. [Introduction](#Introduction)
2. Reconstruct
   1. Noise
   2. Content
   3. Style
3. Visualization
   1. Style
   2. Content
4. 

# Introduction

Style Transfer is the task of composing style from one image which is *style image* over another image which is *content image*. Before doing style transfer using neural network the major limiting factor in this task was feature representation of content and style image for better composition.Lack of such representations thwarted the way to understand the semantics and separation between the two. With the success :heavy_check_mark: of VGG networks on ImageNet Challenge in Object Localization and Object Detection, researchers gave the style transfer a neural approach.



Authors used the feature representations from VGG network to learn high and low level features of both content and style images. Using these explicit information they kept minimizing the loss between content representation and generated image representation using **MSELoss** and between style representation and generated image representation using  **MSELoss of Gram Matrices**. Task of Neural Style Transfer unlike supervised learning doesn't have metric to compare performance of quality of image(s). We are not training model but updating the values of image itself in every iteration using gradient descent such that it match closely with content and style image

# Reconstruct

# Visualization



