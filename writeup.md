## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/car_hist.png
[image3]: ./examples/hog_car.png
[image4]: ./examples/non_car.png
[image5]: ./examples/Non_car_hist.png
[image6]: ./examples/pipeline_test.png
[image7]: ./examples/pipeline_test_heat.png
[image8]: ./examples/pipeline_test_images.png
[image9]: ./examples/windows.png
[video1]: ./project_video.mp4


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Function get_hog_features() is used to extract HOG features from the images. The HOG features are widely use for object detection. HOG decomposes an image into small squared cells, computes an histogram of oriented gradients in each cell, normalizes the result using a block-wise pattern, and return a descriptor for each cell.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of e `vehicle` and `non-vehicle` classes:

#### Car images
![alt text][image1]

#### Non car images
![alt text][image4]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space, Y channel and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

#### Hog Visualization of a car image

![alt text][image3]

#### YCrCb Histogram Visualization of Car and Non Car Images:

#### Cars
![alt text][image2]

#### Non Cars
![alt text][image5]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters like orientation= 9, 11 and 12. I used pixels_per_cell values of (16,16) and (32,32).
Along with various color channels I found that there were various testing accuracy levels. I usually got above 97% accuracy, but with accuracy lesser than 99% there were lot of false predictions. Also different combinations had different no. of features and took different amount of time for training. 

I even acheived 99.44% accuracy with 11 orientations, 32 pixels per cell, YCrCb color space on All channels. But it led to longer training time and I could still see mis classifications. Maybe this was due to the windows getting generated.

I finally settled with the following parameters.

        color_space = 'YCrCb' 
        orient = 12  
        pix_per_cell = 16 
        cell_per_block = 2 
        hog_channel = 0 

It took **57.3 s** to extract all features with training time of **9.36 s** and Test Accuracy of **97.87%**

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the normalized test data. The data was first normalized using StandardScalar()
After that I trained and tested the Linear SVM classifier using the following code:

    from sklearn.svm import LinearSVC
    
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding windows are generated in slide_window() method and are getting searched in search_windows() function.
I decided to do a sliding window search for which I generated the windows of square shapes of various sizes.
I intialized the window size of 72 X 72 and then scaled it up to x1.5, x2.0 and x2.5. I used two rows for each scale.

I used an overlap of 0.9 or 90% on both horizontal axis and the vertical axis.
I selected these values after lot of hits and trials so that the cars in the video start getting detected properly.
Cars far away required smaller window size to get detected. Whereas the cars nearby needed larger windows.
Hence I came up with the aforesaid window sizes.

search_windows() method helps in extracting window features and then making predictions using the classifier. I have not used predict() method instead I have used decision_function() method to get the decision score and based on that I am filtering the responses of the classifier. All values below 0.7 are ignored and rest are predicted as cars.

![alt text][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb, Y-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I was satisfied with the performance of the classifier and hence decided to go with the default parameters. However I augmented the images(flipped images) to make the classifier predict better. Here are some example images:



![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Earlier I did this separately for both sides of the car. Since I observed that the cars on the right side were moving slower than the cars on the left with respect to the our car. As I was also considering the previously predicted frames too, the number of detections were more in the right half than the left half and so a common threshold was not able to work well. 

However this was still satisfactory performance. The reason being at some points no. of positive detections were more and at some places they were less, but the threshold was high after certain no. of frames and so I had to modify it based on the no. of detections in a certain frame. So I implemeted a dynamic threshold which is set to half of the maximum heat value in the heat map adjusted by some arbitrary value.


### Here are some examples and their corresponding heatmaps, output of `scipy.ndimage.measurements.label()` on the integrated heatmap and the resulting bounding boxes

![alt text][image8]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used the LinearSVC with it's default values. I also tried to used SVC with rbf kernel but it took too long for training, hence dropped the Idea. I used the sliding window approach to generate the windows and to make predictions on them. 
I generated windows of varying sizes to detect the vehicles.

Challenges faced:

1. False Positives: I implemented the heat map technique and thresholding for removing the false positives in the video.
2. Flickering Binding boxes: I am considering the previous frames for minimizing flickering and wobbling.
3. Merging two binding boxes: This was done by adjusting the threshold after applying the heat map technique.
4. High Processing time: I skipped alternate frames while creating the output video and retained the last processed frame in its place. Also I was guided by the instructor to focus on the right lane and start searching for cars in the right half only.

Averaging and thresholding are essential for getting a proper output. This solution might fail on other videos as they start searching from a certain position on the image and are dependent upon the view of the camera and inclination of the road. 
I propose to implement a horizon detection technique to find out the position from where to start window search. This will lead to a higher accuracy in detection of vehicles. 

It took me 2 days to fine tune the parameters, still there are some glitches. I ll further try to smoothen the vehicle detection process.
