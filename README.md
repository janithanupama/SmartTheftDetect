# SmartTheftDetect
Real-Time Theft Detection in Supermarkets using Deep Learning and Mobile Edge Computing

# Project Flow

![image](https://github.com/janithanupama/SmartTheftDetect/assets/166873374/2f0617fc-90ec-4ac1-9ed3-54ee9a1ee960)


# Introduction
Theft in supermarkets has been a long-discussed topic which the supermarkets have been trying to address through many means such as CCTV arrays, 24-hour surveillance etc. These methods are prone to failure due to human error in surveillance and human ingenuity where thieves find creative ways to steal. The authors propose and test different approaches to a real-time theft detection system which uses mobile edge computing to process the input video data and send only the relevant processed data back to the cloud or centralized database. The first approach uses long term Recurrent Convolutional Networks where all layers are wrapped by time-distributed layers and sequentially of the data is maintained. As the first approach showed that the data was not enough to train the model. A pre-trained model named VGG16 was used to extract features and was processed to get an output as the second approach. But when compared to the first 
approach its accuracy was lower and was not stable as well. Therefore, in the third approach the accuracy and validation accuracy of the first approach was improved by using image preprocessing prior to training the model and after trying out many techniques such as adaptive mean thresholding, OTSU thresholding, rembg background removal. Thus, it was found that the best results were obtained by the OTSU thresholding method and found that the LRCN approach with OTSU thresholding yielded the best results with highest accuracy and validation accuracy. To show the working of a system in an EDGE server the authors used a raspberry pi 4 model B with 4gb RAM. In Conclusion, the outcome of this research will support the development of more systems using the LRCN approach with OTSU thresholding techniques to help curb theft in supermarkets.

# Steps
1. Pre-processing
2. Training the model
3. Testing the model
4. Implementing the model in a Raspberry Pi 4 (Edge Computer)

# Pre-processing
1. Resizing frames to a small size (1980x1020 -> 224x224)
2. Applying Image 
3. Normalizing Resized frames
4. Applying One Hot Encoding labels
5. Train-Test Split of Data

