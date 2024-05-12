## Main concept of Neural Style Transfer algorithm

![alt text](image.png)



## Calculating Loss function 
![alt text](image-1.png)
![alt text](image-2.png)

The below figure shows different channels of feature maps in any particular layer. At this layer, each channel of this feature map represents the different features present in any image.
-----------------------------------------------------------
![alt text](image-3.png)

Now if we can anyhow find the correlation between these features, we can get the idea of the style as correlation is nothing but the co-occurrence of the features.

Let’s understand this using two vectors a and b. Correlation between these two vectors can be calculated by their dot products. Refer to the image below.

![alt text](image-4.png)

``i.e. the more correlated a and b are, the more is the dot product between them.``

![alt text](image-5.png)

Suppose that the red channel represents the feature of black strips, the yellow channel represents the presence of yellow colour and the green channel represents the presence of white strips. If red channel and yellow channel are fired up with high activation values,i.e. they co-occur then we can say that the image was of a tiger. These two channels will have a higher correlation than that between red and green channels. We know that this co-occurrence can be calculated by calculating the correlation. This correlation of all these channels w.r.t each other is given by the Gram Matrix of an image. We will use the Gram Matrix to measure the degree of correlation between channels which later will act as a measure of the style itself. The formula of the Gram Matrix is given as:

![alt text](image-6.png)
![alt text](image-7.png)

Variation Loss: It wasn’t included originally in the paper. After noticing that reducing only the style and content losses led to highly noisy outputs and to reduce that noise, variation loss was also included in the total_loss of NST. This loss ensures spatial continuity and smoothness in the generated image to avoid noisy and overly pixelated results. This is done by finding the difference between the neighbour pixels.