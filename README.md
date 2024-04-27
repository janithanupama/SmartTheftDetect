# Real-Time Theft Detection in Supermarkets using Deep Learning and Mobile Edge Computing
## Introduction
Theft in supermarkets has been a long-discussed topic which the supermarkets have been trying to address through many means such as CCTV arrays, 24-hour surveillance etc. These methods are prone to failure due to human error in surveillance and human ingenuity where thieves find creative ways to steal. The authors propose and test different approaches to a real-time theft detection system which uses mobile edge computing to process the input video data and send only the relevant processed data back to the cloud or centralized database. The first approach uses long term Recurrent Convolutional Networks where all layers are wrapped by time-distributed layers and sequentially of the data is maintained. As the first approach showed that the data was not enough to train the model. A pre-trained model named VGG16 was used to extract features and was processed to get an output as the second approach. But when compared to the first 
approach its accuracy was lower and was not stable as well. Therefore, in the third approach the accuracy and validation accuracy of the first approach was improved by using image preprocessing prior to training the model and after trying out many techniques such as adaptive mean thresholding, OTSU thresholding, rembg background removal. Thus, it was found that the best results were obtained by the OTSU thresholding method and found that the LRCN approach with OTSU thresholding yielded the best results with highest accuracy and validation accuracy. To show the working of a system in an EDGE server the authors used a raspberry pi 4 model B with 4gb RAM. In Conclusion, the outcome of this research will support the development of more systems using the LRCN approach with OTSU thresholding techniques to help curb theft in supermarkets.

## Project Flow

![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/2f0617fc-90ec-4ac1-9ed3-54ee9a1ee960)

## Steps
1. Pre-processing
2. Training the model
3. Testing the model
4. Implementing the model in a Raspberry Pi 4 (Edge Computer)

## Pre-processing
### 1. Resizing frames to a small size (1980x1020 -> 224x224)
### 2. Using Thresholding Techniques
#### OTSU Thresholding
   ![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/96d81885-f956-4a6e-9912-bc0d327b0a46)
#### Adaptive Mean Thresholding
   ![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/687b68be-8409-4bc3-86a0-14352945e653)
### 3. Using REMBG Background Removal
#### Reference Background Image 
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/8ee3c6d3-1612-40d0-abb0-0e2728b85ab1)
#### Input Image 
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/65314d83-248b-4ca9-bd15-ca2b044522bf)
#### Substracted Image 
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/c1bdc9ee-e33f-4231-9739-96e901019ad9)

### 6. Normalizing Resized frames
### 7. Applying One Hot Encoding labels
### 8. Train-Test Split of Data

## Training
1. Convolution 2D Layers - Extract spatial features from the frames in the video and creating an activation map.
2. LSTM Layers - Use features extracted by Convolution layers, to predict the action being performed in the video.
3. Dropout Layers - Used to prevent overfitting of the model (Reduce the dimensions of the frames).
4. Flatten Layer - To flatten features extracted by the Conv2D layers. Converting the data into a 1-dimensional array for inputting it to the next layer.
5. Middle Layer Activation Function - Relu
6. Final Dense Layer Activation Function - SoftMax

## Results
## 1. CLSTM/LRCN Approach
### Accuracy vs Validation Accuracy
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/052e8718-e2af-42eb-be02-709ede703f61)

### Loss vs Validation Loss
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/a23f21f7-4bd6-41e9-a2bb-0cf8fac8e020)

## 2. VGG16 (Transfer Learning)
### Accuracy vs Validation Accuracy
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/9c4697ba-0a08-4b36-aa67-e1a373c1e5df)

### Loss vs Validation Loss
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/5a4caf4a-23f8-4a1d-baf4-fdeb52e92afd)

## Final Implementation
![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/6b08a565-374b-42ba-a4b2-4a94c433dc6d)


