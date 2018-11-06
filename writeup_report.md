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
8. Different Object Detections.
9. Future Enhancement.  

[//]: # (Image References)
[image1]: ./images/Figure_1.jpg
[image2]: ./images/Figure_2.png
[image3]: ./images/Figure_3.png
[image4]: ./images/Figure_4.png
[image5]: ./images/Figure_5.png
[image6]: ./images/Figure_6.png
[image7]: ./images/Figure_7.png
 
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
refered to as "dense" layer. In this project we didn't use fully connected layer since it doesn't give spatial information of the pixels
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
As shown above, our model consists of 3 successive encoder layers, followed by a 1X1 convolution layer. Then, 3 decoders for upsampling
(including concatination for skip connections from previous layer). Finally, a softmax function is applied to get the output in the desired dimension. **Note** Each decoder layer contains 2 successive convolutional layers but with stride=1 so that they doesn't downsample the result of the deoder again. **Note** Each encoder is called with stride=2, i.e. the height and width of the resulting image both are half of those of input image.

#### Encoder code 
```
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer,filters,strides)
    return output_layer
```
#### Decoder code 
```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    output_layer = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output_layer = layers.concatenate([output_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output_layer,filters)
    output_layer = separable_conv2d_batchnorm(output_layer,filters)
    return output_layer
```
#### The whole model    
```
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    x1 = encoder_block(inputs,32,2)
    x2 = encoder_block(x1,64,2)
    x3 = encoder_block(x2,128,2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    x4 = conv2d_batchnorm(x3,256,1,1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    x5 = decoder_block(x4,x2,128)
    x6 = decoder_block(x5,x1,64)
    x = decoder_block(x6,inputs,32)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## 5. Model Training:
In our training we didn't choose a constant values for parameters. By try and error we found that small batch size with larger number for spoch steps gives better accuracy in fewer number of epochs. But we didn't use a constant learning rate. 
We used constant values for batch size, steps per epoch, and steps per evaluation: 
```
batch_size = 22
steps_per_epoch = 188
validation_steps = 54 
```
To decrease the number of epochs for training and achieve the required accuracy we started with relatively large learning rate. After the training has finished with this learning rate we repeated the training on the model resulted from the previous time, but with lower value for learning rate and lower number of epochs and so on. The total number of epochs required was 64 spochs as follows:
```
1. num_epochs = 35 with learning_rate = 0.007.
2. num_epochs = 10 with learning_rate = 0.002.
3. num_epochs = 15 with learning_rate = 0.001.
4. num_epochs = 1  with learning_rate = 0.007 (This step is required to avoid overfitting and repeated about 4 times)
```
## 6. Model Prediction:
The weights of the trained model are saved in `model_weights.h5` file. When we use this model for prediction, it is tested 
for three cases: following the target, patrol without target, and patrol with target. The code for testing the three cases is
found below with images showing the success of each case.
#### Follwing the target
```
# images while following the target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```
![alt_text][image2]
![alt_text][image3]
#### Patrol without target
```
# images while at patrol without target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
 
```
![alt_text][image4]
![alt_text][image5]
#### Patrol with target
```
# images while at patrol with target
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```
![alt_text][image6]
![alt_text][image7]

## 7. Model Evaluation:
##### According to the prediction step using our trained model we can calculate scores of different tests.

Scores for while following the target:
```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9964661320129408
average intersection over union for other people is 0.40119914857744604
average intersection over union for the hero is 0.9239851423095123
number true positives: 539, number false positives: 0, number false negatives: 0
```
Scores for while patroling whithout target in the scene:
```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9892102459455289
average intersection over union for other people is 0.7765001088344552
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 66, number false negatives: 0
```
Scores for while patroling with  target in the scene:
```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9968986758076048
average intersection over union for other people is 0.48017020988598014
average intersection over union for the hero is 0.23683195381976854
number true positives: 128, number false positives: 4, number false negatives: 173
```
The Final evaluation is calculated using IOU (Intersection Over Union) which is the ratio between two components: 
(pixels that are a part of a class AND classified as a part of that class i.e true positive) and (pixels that are part of a class OR pixels that classified to be part of that class i.e true poisive + false positive).

##### Total score 
Sum all the true positives, etc from the three datasets to get a weight for the score 
```
weight = true_pos/(true_pos+false_neg+false_pos)

weight = 0.7329670329670329
```
The IOU is the average between IOU1 and IOU3 (i.e. execlude testcases where there is no target in the scene) 
```
final_IOU = 0.580408548065
```
Then the final grade score calculated as: final_score = final_IoU * weight
```
final_score = 0.425420331384
```
This score meet the required specs (>0.4)

## 8. Different Object Detections:
In this project we limited our work to follow a human target. So, a different object like dog, car or cat can't be detected and consequently can't be followed since the model have not been trained on different objects to detect. To make our model able to detect different object, we must include these objects in training data with the right labeled output. 

## 9. Future Enhancement: 
In this project we reached an accuracy of IOU = 0.425 which is required. For this model to be reliable, it is desired to increase its accuracy. We can do this by means of three directions. First one, is to partially train the network i.e. train the encoders to get a good classifier then train the decoder. Second one, is to make a good use of the idea of different learning rate i.e. design a learning 
rate function which is exponentially decreasing with epochs. We think that this function will lead to faster convergence. The last direction, is to capture more images and increase the size of training and validation data to capture more and more data. 
