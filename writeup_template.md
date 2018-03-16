## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
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

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space, Y channel and HOG parameters of `orientations=12`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters like orientation= 9, 11 and 12. I used pixels_per_cell values of (16,16) and (32,32).
Along with various color channels I found that there were various testing accuracy levels. I usually got above 97% accuracy, but with accuracy lesser than 99% there were lot of false predictions. Also different combinations had different no. of features and took different amount of time for training. 

I even acheived 99.44% accuracy with 11 orientations, 32 pixels per cell, YCrCb color space on All channels. But it led to longer training time and I could still see mis classifications. Maybe this was due to the windows getting generated.

I finally settled with the following parameters.
color_space = 'YCrCb' *# Can be RGB, HSV, LUV, HLS, YUV, YCrCb*
orient = 12  *# HOG orientations*
pix_per_cell = 16 *# HOG pixels per cell*
cell_per_block = 2 *# HOG cells per block*
hog_channel = 0 *# channel*

It took **56.8s** to extract all features with training time of **9.01s** and Test Accuracy of **0.9821**


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the normalized test data. The data was first normalized using StandardScalar()
After that I trained and tested the Linear SVM classifier using the following code:

    from sklearn.svm import LinearSVC
    
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to do a sliding window search for which I generated the windows of square shapes of various sizes.
I intialized the window size of 64 X 64 and then scaled it up to x1.5, x2.0 and x2.5. I used two rows of each scale.
And started the each new window size, 20px below the previous window's starting position.

I used an overlap of 0.9 or 90% on the horizontal axis and 0.8 or 80% on the vertical axis.
I selected these values after lot of hits and trials so that the cars in the video start getting detected properly.
Cars far away required smaller window size to get detected. Whereas the cars nearby needed larger windows.
Hence I came up with the aforesaid window sizes.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb, Y-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:



![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I did this separately for both sides of the car. Since I observed that the cars on the right side were moving slower than the cars on the left with respect to the our car. As I was also considering the preciously predicted frames too, the number of detections were more in the right half than the left half and so a common threshold was not able to work well. 

Since I am taking past predicted values too for the heat map, I implemeted a dynamic threshold which increases as the number of frames accumulate. I implemented a threshold of minimum(No of previously detected frames, max frames over which averaging is being done) multiplied by an arbitrary number that suits the detection in the video.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used the LinearSVC with it's default values. I also tried to used SVC with rbf kernel but it took too long for training, hence dropped the Idea. I used the sliding window approach to generate the windows and to make predictions on them. 
I generated windows of varying sizes to detect the vehicles.

Challenges faced:

1. False Positives: I implemented the heat map technique and thresholding for removing the false positives in the video.
2. Flickering Binding boxes: I am considering the previous frames for minimizing flickering and wobbling.
3. Merging two binding boxes: This was done by adjusting the threshold after applying the heat map technique.

