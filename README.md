# retinal-blood-vessel-segmentation-via-machine-learning






**Introduction** 

**What is Retinal Blood Vessel Segmentation?**

Retinal blood vessels are the only deep vessels that can be observed in human body and directly reflect the omen of some cardiovascular diseases and severity of diabetic retinopathy. Therefore it is essential for eye doctors to identify the retinal blood vessels from retinal images to diagnose eye diseases. The pixelwise classification of vessels from retinal images is called Retinal Blood Vessel Segmentation.

Below is one example of an image of eyeball (training image for the machine learning model):
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/21_training.png)

Below is one example of an Retinal Blood Vessel Segmented image (ground truth for training):
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/21_manual1.gif)




**Why Machine Learning?**

Retinal blood vessel segmentation is very time-comsuming (as each pixel needs to be classified) to perform manually and only experienced eye doctors can accurately perform the task. Therefore a Machine Learning approach can extensively expediate the segmentation process while achive higher accuracy than manual segmentation. 



**High Lights** 

• Given 40 patients’ eyeball images, predicted pixel wise retinal blood vessels in eyeball images

• Built U-net Convolutional Neural Network (CNN) model (Python, Keras) for pixel wise classification

• Preprocessed and augmented the 20 training images to obtain 400 images by mask, normalization, shearing 

• Achieved 96% (IoU) accuracy on 20 patients’ retinal blood vessel test images


My U-Net CNN architechture achieves 96% IoU accuracy, which is higher than the average of a manual segmentation. So Machine Learning algorithm does win over experienced eye doctors in both speed and accuracy!

Below is the plot for the loss and IoU accuracy of the 40 test images and 100 training images:
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/metric.jpg)



**Data** 

I used the Digital Retinal Images for Retinal Extraction database, which is a database of manual segmentations done on the retinal vasculature (https://www.isi.uu.nl/Research/Databases/DRIVE/index.html). The data set contains 40 manually labeled images, divided into a training and a test set, both containing 20 images. For the training images, a single manual segmentation of the vasculature is available. For the test images, two manual segmentations are available; one is used as standard, the other one then can be used to compare computer generated segmentations with those of an independent human observer. For purposes of this project, I only consider the manual segmentations as ground truths. 

In this study, I focus on how to identify the blood vessels from retinal images using automatic approaches. The use of automatic diagnostic tools for retinal images will save time and improve the accuracy compared to manual diagnosis, especially when faced with small vessels that are hard for human identification. There are numerous supervised and unsupervised machine learning models that can be trained to identify the retinal images at present. In this article, I discuss supervised segmentation approaches, such as deep, convolutional neural networks and how they are used to extract features from the pre-processed image. Specifically, this work focuses on using a pooling-upsampling approach called U-Net




**U-Net Model**

I use the U-Net "fully connected" convolutional network to classify every pixel of the input image. I have attempted to run the full, 1/2 , and 1/4 U-net models  with respect to the number of 2D convolution at each phase of the architecture.   The first model used in our work is the original  10 layer model with 64 filters in the first layer. This model itself contains sequential two-dimensional (2D) convolutions, followed by max-pooling in the "down-side" of the architecture, and copy-and-crop in the horizontal direction. Both sides of the architecture contain only ReLu activation functions, which helps eliminate the possibility of gradient blow-up or vanishing.  An additional set of up-sampling and 2D convolutions follows (i.e. the up-side), creating the iconic U shape of the architecture. After the down  and up side's of the architecture have finished, each pixel is classified into two classes: either the foreground (the retinal blood vessels) or the background (every other pixel). This is done by making the very last convolutional layer a soft-max function. 

One draw-back of the original full u-net model is that it uses a large number of parameters and computationally expensive.  To solve this problem, I first try to  reduce the number of layers of the original U-Net. This reduced layer model has fewer layers than the original model. However, reducing the number of layer from 10 to 8 does not solve the problem of the original model and the reduced model still gives too many parameters for available computing resources. Therefore, instead of reducing the number of layers, I reduce the number of filters in each 2D convolution layer. As a result, the 1/2 , and 1/4 U-net models both predict pretty accurate results within a reasonable period of  time.

Since I use valid padding in all convolutional layers, the output size of the u-net models is smaller than the input size. However, in order to compare the images after training with the  truth images, I must ensure that the output size of the u-net model equals that  of the truth. I can accomplish this by padding the original image. For example, given images with red, green, and blue (RGB) components of size 532*532*3  and truth images with size 532*532*1, I can use zero-padding to make the input size of the u-net model equals 716*716*3. The resulted output size would then be 532*532*2.

Full U-Net with 16 filters in conv 1, also referred as 1/4 model, model 2, or  16 2D convolution first  layer  model:
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/pic1.png)




Reduced layer U-Net model. In contrast to the original architecture, the Reduced U-Net only has three (instead of four) successive 2D convolutions on the up and down sides:
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/u-net.png)





**Three attemps to augment data**

The image-size and the ground truth are 2D images of size 584*565*3 and 584*565, respectively. In order to fit the proper size of input for u-net, we need to crop the images to be square, making the sizes  584*565*3 for the image and 584*565*1 for the ground truth. Similarly, we crop the testing images to the same dimensions. Using 20 images for training the architecture is not ideal, as the machine learning parameters cannot be learned with such a small data set. To combat this, we considered several different pre-processing steps.

First, we standardize each images so that each of the RGB components are scaled from [0,255] to [0,1]. This assists in creating larger contrast between intensity regions. In particular, the retinal images provided are a reddish hue and can be contrasted in a better manner after standardization.

Our next procedure is to increase the size of the training samples. Several attempts were made to enhance the dataset available for training. Initially, we cut the images into 4 equal quadrants. This produces 80 training segments from the original 20. We consider symmetric padding for this set of data, since the quadrants are not in the proper size for fitting the reduced U-Net model. Next we apply shearing and rotation in different degree for those quadrants to generate more training and corresponding truth images. 


The  image prepossessing is  described in the three figures below:
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/pro1.png)


Ten data augmentations were applied to each quadrant to increase the number of training images and truths to 800. However, the model tended to converge rather quickly (within one epoch) to a accuracy around 95%. The accuracy metric is telling of the goodness of the machine learning prediction; however, 95% of the quadrants used consisted of one class. This lead to data imbalance and an inability to identify the vessels in the images. Model predictions only contained black pixels of the background, since the background is the dominance of the truth images. I believe that the reduced U-Net is training, but cannot associate spatial context with the symmetric padding that was applied. It could also be that by removing a set of 2D convolutions and pooling, local information about the image was not used in training. Hence, I changed the symmetric padding to zero padding, which is adding zeros around the images and then cut into quadrants. Unfortunately, the model still predicts the all background. To assist with the inconsistency of the accuracy metric, I consider the Dice Coefficient, which compares the cardinality of the union of the two sets to the sum of the cardinalities. A Dice score of 1 indicates perfect matching between two sets, and serves as an appropriate metric for image segmentation problems.

![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/pro2.png)

I ultimately changed back to the full U-Net model, which contains 4 convolutional layers with pooling, a single convolution layer, and 4 convolutional layers with upsampling. I tested different numbers of convolutional operations at each layer to see which performed best, and chose to test the standard U-net (64 convolutional layers at the start, with doubling of layers after each pooling) versus two reduced models (32 and 16 convolutional layers at the start with doubling after each pooling), denoted as model 1 and model 2, respectively. I preprocess the data by standardizing the RGB components, and adding the zero padding around the entire image to make the image size to be $716\times 716\times 3$. To make the training more efficient, I use the masks provided in the data set to distinguish the center of our training images. I also use different rotations and shearing in order to increase sample size. 

![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/pro3.png)




**Results**

Each model was provided with all 20 images under 40 random augmentations that included rotation and shearing. Due to the computational complexity of the U-Net architecture, I only ran 50 epochs for the 64 and 32 first layer 2D convolution models, and computed 100 epochs for the smallest, 16 2D convolution first layer model. The abundance of parameters in the two larger models made training an issue, since computational resources were limited. The original U-Net model was unable to train on available CPU machines, hence I focused our attention on the two reduced U-Net models. We will denote the 32 and 16 2D convolutional first layer models as model 1 and model 2, respectively.

After training both networks, I saw comparable results. The maximum dice coefficient in the set was 0.9690, while the minimum was 0.9201. Overall, the two models predict within this range for the testing images. However, model 1 has only trained for 50 epochs compared to model 2 which has trained for 100 epochs. Thus, given more training time, I might expect model 1 to outperform model 2. This is intuitive, as model 1 contains more model parameters that can better model the ground truth observed. Results for model 2 are the only results shown for all 20 testing and training images, but are nearly identical visually to model 1


One of the test images: The blue line is the predited vessel, the yellow line is the ground truth

![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/test_1_model2.png)


Fully trained training images 
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/Training_all20.png)

Fully trained test images 
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/Testing_all20.png)



**Reference**
J.J. Staal, M.D. Abramoff, M. Niemeijer, M.A. Viergever, B. van Ginneken, "Ridge based vessel segmentation in color images of the retina", IEEE Transactions on Medical Imaging, 2004, vol. 23, pp. 501-509.

Ronneberger, Olaf, et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015, pp. 1–8, doi:10.1007/978-3-319-24574-4_28.

Çiçek, Özgün, et al. “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.” Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 9901 LNCS, 2016, pp. 424–32, doi:10.1007/978-3-319-46723-8_49.



