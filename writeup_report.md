# Project: Follow Me - Deep Learning 
Before starting deeplearning description, we build the project as explained in RoboND-DeepLearning-Project readme file 
---
**Then our work main steps are:**
1. Collecting Data. 
2. FCN Components.
3. Network Architecture.
4. Model Building.
5. Model Training.
6. Model Prediction.
7. Model Evaluation.
8. Future Enhancement.  

[//]: # (Image References)
[image1]: ./images/Figure_1.png
 
## 1. Collecting Data: 
In our work on this project we didn't need additional data for training the network to meet the reqired specifications.
So, we only used the given data for training and validation.

## 2. FCN Components:
Fully Convolutional Networks is used for semantic segmentaion of the scene to detect required objects and their locations. 
This is different to the classical convolution networks which composed only of convolutional layers (encoders) and are capable of 
classification tasks. The FCN is basically a concatination of consequent stages: 
#### Encoder 
Encoder is the convolutional layer in the FCN. You Apply the kxk kernel to map the input image to a deeper output enabling the extraction
of more information from the image. The output of an encoder may be passed as an input to a deeper convolutional layer for more information 
extraction. 

#### Fully Connected Layer 
Fully Connected Layer is a classical NN layer where all input neurons are connected to output neurons resulting in a large number of weights
refered to as "dense" layer. In this project we didn't use fully connected layer since it doesn't give spation information of the pixels
which is required in object tracking (It is suitable in classification tasks).
#### 1x1 convolution  
1x1 convolution is applied to the output of a convolution layer. The aim of 1x1 convolution is encoding spatial information and (may be) reducing dimensionality of a layer. 
By definition, compared to fully connected layer, 1x1 convolution added advantage of working with different input sizes during prediction. 
In this project we add a 1x1 convolution layer after encoder layers.    
#### Decoder  
Decoder is the transposed convolution layer in the FCN. It is rseponsible for upsampling the previous layer to desire resolution. 
The Decoder is an important part that make difference detween FCN and classical CNN as it enbale determination of the position of the object to
be detected in the image. Decoder layer may take data from previous layer to upsample it or take data from input layer with the same resolution of the decoder output. 
This process is called skip connections and enables passing certain information from input to the output stage. We use both upsampling and skip connection operartions
in each decoder of the network.

## 3. Network Architecture:
In this project we adobt the architecture shown below. This architecture is the one gave us required accuracy. If we use less layers 
than three for encoders and decoders we doesn't get what we want. (Smaller netwok may work if we trained it more enough say > 100 epochs which is not our design).
![alt_text][image1]

## 4. Model Building:

## 5. Model Training:

## 6. Model Prediction:

## 7. Model Evaluation:

## 8. Future Enhancement:
