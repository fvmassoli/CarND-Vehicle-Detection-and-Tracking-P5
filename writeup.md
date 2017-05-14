**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_image.jpg
[image2]: ./output_images/hog_vehicle_image.jpg
[image3]: ./output_images/non_vehicle_image.jpg
[image4]: ./output_images/hog_non_vehicle_image.jpg
[image5]: ./output_images/sliding_window_original_image.jpg
[image6]: ./output_images/sliding_window_grid_image.jpg
[image7]: ./output_images/reduced_false_positive_image.jpg
[image8]: ./output_images/heatmap_image.jpg
[image9]: ./output_images/reduced_false_from_heatmap.jpg
[video1]: ./test_video_output.mp4
[video2]: ./test_video_output_ave.mp4
[video3]: ./project_video_output.mp4
[video4]: ./project_video_output_ave.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Histogram of Oriented Gradients (HOG)

#### 1-2. Explain how (and identify where in your code) you extracted HOG features from the training images.

In order to properly extract HOG features I did several test. I got the best results by using all the channels in the YUV color space and following the suggestions from the lessons I used 9 HOG orientations, 8 pixels per cell and 2 cells per block. The chosen value for the parameter are shown in the cell 3 of the IPython notebook. 

From cell 4 to cell 7 I implemented the code that extracts and normalize the feature vector of the images.
The FeatureExtractor.py file contains the FeatureExtractor class. The class is used to extract features from images. In particular the  main method is get_features(). It relies upon calls to other three methods of the class: get_hog_features(), get_color_hist() and get_bin_spatial(). The get_features() then combines the return values from the previous three methods and returns the frature vector.

In order to avoid some bias fue to a major contribution from one of the previous components, ater the features concatenations I normalized the features and used the normalized feature vector in the remaining part of the project.

Below there is an example of the HOG images extracted from a vehicle and non-vehicle input images.
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the cells 12 and 13 of the IPython notebook there is the implementation of the training of the classifier. 
The training and test sets contains 70% and 30%, respectively, of the input images. The final accuravy of the classifier is 99%.


### Sliding Window Search

#### 1-2. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
#### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to look into the portion of the image where a car is expexted to be I didn't consider the top 400 pixels and left 300 pixels of the image. That trick also makes the algorithm faster since it deals with fewer windows.
Once I had the image ROI I considered all the possible 64x64 windows with 32x32 stride.

Below there is an example.

![alt text][image5]
![alt text][image6]


Then, for each window in above image, we predict if it is a vehicle image or not. If it is a vehicle image, we added to our list and draw end result on the image.

The results are shown in figures below.

![alt text][image7]


In order to decrease the number of false positives, I analysed the images using heat maps. The procedure is pretty straightforward. First I creaate a zero mask for the image. Then for each box found in the previous step I increased by one the value of the pixels in the mask. The final step is to apply a threshold to the mask. This procedure really helps the algortihm in reducing the probability of false positive.

The final result is shown in picture below. In order to workout the previous procedure, the code is implemented in the add_heat() and apply_threshold() methods shows these procedures.

Below there is an example of the previous analysis.

![alt text][image8]
![alt text][image9]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to the test video result](./test_video_output.mp4)

Here's a [link to the project video result](./project_video_output.mp4)


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the final step of the project I tried to average the search for cars by considering 20 images at a time. This is a strategy that can be improved even more by choosing the best number of frames to consider or by considering different thresholds vbalue for the heatmaps. Indeed, by using 20 frames some false positives are introduced into the output video perhaps due to shadowing effects or other image features that when summed up go beyond the threshold and give rise to a false positive.

The output from the averaged pipeline are showed below.

Here's a [link to the test video result](./test_video_output_ave.mp4)

Here's a [link to the project video result](./project_video_output_ave.mp4)






