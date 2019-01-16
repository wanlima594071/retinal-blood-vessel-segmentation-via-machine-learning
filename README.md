# retinal-blood-vessel-segmentation-via-machine-learning

Wanli Ma, Mitchel Colebank, Emily Zhang

Data 

We used the Digital Retinal Images for Retinal Extraction (\hyperlink{https://www.isi.uu.nl/Research/Databases/DRIVE/index.html}{DRIVE}) database, which is a database of manual segmentations done on the retinal vasculature. The data set contains 40 manually labeled images, divided into a training and a test set, both containing 20 images. For the training images, a single manual segmentation of the vasculature is available. For the test images, two manual segmentations are available; one is used as standard, the other one then can be used to compare computer generated segmentations with those of an independent human observer. For purposes of this project, we only consider the manual segmentations as ground truths. 

Reference: J.J.  Staal,  M.D.  Abramoff,  M.  Niemeijer,  M.A.  Viergever,  and  B.  van  Ginneken.   Ridge  based  vesselsegmentation in color images of the retina.IEEE Transactions on Medical Imaging, 23(4):501–509, 200

One example of training images (image of eyeball):
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/33_training.tif)

Ground Truth images for training images (manual segmentation of the vasculature):
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/all_predictions.png)

In this study, we focus on how to identify the blood vessels from retinal images using automatic approaches. The use of automatic diagnostic tools for retinal images will save time and improve the accuracy compared to manual diagnosis, especially when faced with small vessels that are hard for human identification. There are numerous supervised and unsupervised machine learning models that can be trained to identify the retinal images at present. In this article, we discuss supervised segmentation approaches, such as deep, convolutional neural networks and how they are used to extract features from the pre-processed image. Specifically, this work focuses on using a pooling-upsampling approach called U-Net


U-Net Model
We use the U-Net "fully connected" convolutional network to classify every pixel of the input image. We have attempted to run the full, 1/2 , and 1/4 U-net models  with respect to the number of 2D convolution at each phase of the architecture.   The first model used in our work is the original  10 layer model with 64 filters in the first layer. This model itself contains sequential two-dimensional (2D) convolutions, followed by max-pooling in the "down-side" of the architecture, and copy-and-crop in the horizontal direction. Both sides of the architecture contain only ``ReLu" activation functions, which helps eliminate the possibility of gradient blow-up or vanishing.  An additional set of up-sampling and 2D convolutions follows (i.e. the ``up-side"), creating the iconic ``U" shape of the architecture. After the down  and up side's of the architecture have finished, each pixel is classified into two classes: either the foreground (the retinal blood vessels) or the background (every other pixel). This is done by making the very last convolutional layer a soft-max function. 

One draw-back of the original full u-net model is that it uses a large number of parameters and computationally expensive.  To solve this problem, we first try to  reduce the number of layers of the original U-Net. This reduced layer model has fewer layers than the original model. However, reducing the number of layer from 10 to 8 does not solve the problem of the original model and the reduced model still gives too many parameters for available computing resources. Therefore, instead of reducing the number of layers, we reduce the number of filters in each 2D convolution layer. As a result, the 1/2 , and 1/4 U-net models both predict pretty accurate results within a reasonable period of  time.

Since we use valid padding in all convolutional layers, the output size of the u-net models is smaller than the input size. However, in order to compare the images after training with the  truth images, we must ensure that the output size of the u-net model equals that  of the truth. We can accomplish this by padding the original image. For example, given images with red, green, and blue (RGB) components of size 532*532*3  and truth images with size 532*532*1, we can use zero-padding to make the input size of the u-net model equals 716*716*3. The resulted output size would then be 532*532*2.

Fully trained training images 
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/Training_all20.png)

Fully trained test images 
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/Testing_all20.png)


This is one of our results:
![github-small](https://github.com/wanlima594071/retinal-blood-vessel-segmentation-via-machine-learning/blob/master/test_1_model2.png)






