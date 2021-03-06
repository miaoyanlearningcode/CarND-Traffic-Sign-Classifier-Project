#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output_Result/histogram_origin.png "histogram1"
[image2]: ./Output_Result/histogram_fakeData.png "histogram2"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./data_web_test/canstock14957677.jpg "Traffic Sign 1"
[image5]: ./data_web_test/different-traffic-signs-in-use-in-germany-c19aj4.jpg "Traffic Sign 2"
[image6]: ./data_web_test/german2.jpg "Traffic Sign 3"
[image7]: ./data_web_test/german1.jpg "Traffic Sign 4"
[image8]: ./data_web_test/100_1607.jpg "Traffic Sign 5"
[image9]: ./data_web_test/german3.jpg "Traffic Sign 6"
[image10]: ./data_web_test/ped.jpg "Traffic Sign 8"
[image11]: ./data_web_test/german4.jpg "Traffic Sign 9"
[image12]: ./Output_Result/normalizedImg.png "Normalization"
[image13]: ./Output_Result/crop.png "crop"
[image14]: ./Output_Result/43Classes.png "dataVis"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/miaoyanlearningcode/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 

![alt_text][image14]

Here is a bar chart showing how the number of each class.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

As a first step, I generated the fake data so that the numbers of different labels are close. What I did is that I found the label of maxmium number. For each label, compare the number of this labe to the maximum number to find how many data I should genereate. For example, if the maximum number is 10 times as many as the number of label 0, I need generate 9 times more data for label 0. What I did was roataing the images label 0. Roation angle is from -15 degree to degree.  

Here is the histogram of numbers of different classes after generating fake data:

![alt text][image2]
After data augment, the total number of data including training and validation is 83007 and we split it into training set and validation set. 
Training set has 66405 data.
Validation set has 16602.


As a second step, I normalized the image data. Images taken in different light conditions will lead to different RGB values, which will make troubles to train the weights of CNN layers. Normalization will make the image values between -1 and 1. 
Here is one example of normalized Image:

![alt text][image12]

The normaliztion process is as following:

1. calculate the means and stand deviation of each channel for each image
2. (image - mean) / std


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the tenth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by combining the training data and valid data together, generating the data like I described before and split the data into train data and valid data again using function train_test_split from sklearn.model_selection.

My final training set had 66405 number of images. My validation set and test set had 16602 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| relu | |
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x8 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x16    |
| relu | |
| Max pooling			| 2x2 strid, valid padding, output 6x6x16
| Fully connected		| output 128      									|
|relu||
| Fully connected | output 43				
| softmax| output : logits|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 12th cell of the ipython notebook. 

To train the model, I used an AdamOptimizer and set batch size as 256, number of epoches as 50 and learning rate as 0.0001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 99.40% 
* test set accuracy of 93.25%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

	The first architecture I tried is LeNet. Because we learned LeNet in the class and it is easy for us to implement to get the initial result and also LeNet is the classic architecture of CNN which is worthwhile for me to try. 
* What were some problems with the initial architecture?
	
    First, the output layer of LeNet is 10, but german traffic signs have 43 classes. 
    Second, the valid accuracy and test accuracy is around 90%, whic is not good enough.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

	1. Change the output layer to 43 instead 10.
	2. To increase the accuracy, I used dropout to avoid overfitting. To keep more information, I changed the convolution layers. The first convolution filter is 5*5*8, and second one is 3*3*16. 
* Which parameters were tuned? How were they adjusted and why?
	
    1. learning rate. If the accuracy starts decreasing and increasing randomly after some epoches, it means learning rate is too large.
    2. keep probability of drop: 0.5 is the typical value for training and 1.0 for testing.
    3. Epoches: If learning rate is set to be very small, the epoches should increase so that the loss function can reach the minimum.
    4. batch_size: Since the number of classes is 43. If we still use 128 as batch_size, the training will take longer.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	
    CNN working well is because the convolution layers can grab the key features from images and the features could be more complicated as adding more layers. The dropout can avoid overfitting. So the test accuracy can increase. 


If a well known architecture was chosen:
* What architecture was chosen?
	
    LeNet
* Why did you believe it would be relevant to the traffic sign application?

	Because LeNet works well in classifying numbers. It can grab the key features of images and classify them. And traffic sign classificaion is the similar problems but with more data and classes. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	
    Training Accuracy:
    Validation Accuracy:
    Test Accuracy:
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight images of german traffic signs and each has a sign to detect except the second one which three signs were cropped. So there are 10 traffic signs for us to test. 

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt_text][image9]
![alt_text][image10] ![alt_text][image11]

Here are the signs cropped from the images above:
![alt_text][image13]

I think the pedestrian sign is the hardest one to predict. There are two reasons: first, the pedestrian sign is taken from the angle that is different from most images; second, during the cropping process, I could not find a good way to crop it to square shape. 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h) 									| 
| Go straight or right     			| Go straight or right 										|
| No vehicles					| No vehicles											|
| No entry	      		| No entry					 				|
| Yield			| Yield      							|
|	General caution     |	General caution
|Right-of-way at the next intersection|Right-of-way at the next intersection|
|Priority road|Priority road|
|Road narrows on the right|Children crossing|
|No entry|No entry|


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This is close to the accuracy I got from the test set. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit(30 km/h) (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99998689e-01  | Speed limit (30km/h) | 
| 1.02727552e-06  | Roundabout mandatory |
| 2.48929041e-07  | Speed limit (50km/h) |
| 3.21363913e-09  | Speed limit (80km/h) |
| 1.05783479e-10  | Speed limit (60km/h) |


For the second image, the model is relatively sure that this is a go straight or right (probability of 0.996), and the image does contain a go straight or right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99600470e-01      | Go straight or right  | 
|  3.97549651e-04     | Roundabout mandatory   |
|   9.74083946e-07    | Go straight or left    |
|  6.75855972e-07     | Keep right            |
|2.92002625e-07       | Priority road         |  			


For the third image, the model is relatively sure that this is a No Vehicles (probability of 0.9987), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99879122e-01     | No vehicles          | 
| 1.04565108e-04     | No passing           |
| 1.60360214e-05     | Yield		        |
| 2.57449500e-07     | Speed limit (30km/h) |
| 1.64997491e-08     | Traffic signals    |


For the fourth image, the model is relatively sure that this is a No Entry (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999404e-01  	   | No entry        | 
| 6.16680666e-07       | Stop	         |
|    5.10661595e-14    | No vehicles     |
|     3.60137843e-14   | Bicycles crossing     |
|   2.50287089e-14     |  Yield  |

For the fifth image,  the model is relatively sure that this is a Yield (probability of closing to 1), and the image does contain a stop sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	1.00000000e+00     	| Yield         | 
|   8.14774897e-12     	| Ahead only	|
|   6.98552154e-12		| No passing    |
|  	8.16244424e-13      | Keep right	|
|	5.38518665e-13      |Priority road  |

For the sixth image,  the model is relatively sure that this is a General Caution (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	| 	Prediction        					| 
|:---------------------:|:---------------------------------------------:| 
| 	9.99994516e-01 	|  General caution   		| 
| 	5.42405996e-06  | Pedestrians				|
|	2.22703700e-09  | Traffic signals			|
|	5.44925173e-12  | Road narrows on the right	|		
| 	2.09899914e-12  | Right-of-way at the next intersection|

For the seventh image, the model is relatively sure that this is a Right-of-way at the next intersection (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	9.99992609e-01   | Right-of-way at the next intersection  | 
|	6.09703875e-06   |General caution 		|
| 	1.25875556e-06   | Speed limit (30km/h)	|
|   2.94801988e-10   | Pedestrians 			|
|	2.85679036e-10   | Double curve  		|

For the eighth image, the model is relatively sure that this is a priority road (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	1.00000000e+00   | Priority road  | 
|  	2.59046663e-14 	 | Stop 		|
|	5.11301698e-16   | No entry		|
|  	5.38284940e-19	 | No passing for vehicles over 3.5 metric tons|
| 	7.50166173e-24   | End of no passing by vehicles over 3.5 metric tons    |

For the ninth image, compare with the other signs, the model is relatively not sure that this is a Children crossing (probability of 0.72, compared with other probabilities of 0.999), and the image does contain a stop sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	0.786551058  | Road narrows on the right| 
|  	0.207940593   | General caution	|
|   0.00483439211  | Traffic signals	|
|   3.68204725e-04	 | Road work		|
| 	8.45819086e-05   | Speed limit (70km/h)	|

For the tenth image, the model is relatively sure that this is a No Entry (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 	1.00000000e+00 | No entry           | 
|	1.07294895e-09 | Stop               |
|	9.05864574e-16 | No vehicles        |				
|   8.27528267e-16 | Traffic signals	|
| 	4.68769262e-16 | Bicycles crossing  |