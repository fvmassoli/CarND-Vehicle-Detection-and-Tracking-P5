from skimage.feature import hog
from scipy.ndimage.measurements import label
import numpy as np
import cv2

class FeatureExtractor(object):

    def get_features(self, image, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

        features = []
        file_features = []

        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if spatial_feat == True:
            spatial_features = FeatureExtractor.bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            hog_features = FeatureExtractor.get_hog_features(feature_image, orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
        return features



    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
        hog_features = []
        hog_img = np.zeros_like(img)
        for channel in  range(img.shape[2]):
            if vis == True:
                features, hog_image = hog(img[:,:, channel], orientations=orient,
                                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                                              cells_per_block=(cell_per_block, cell_per_block),
                                              transform_sqrt=True,
                                              visualise=True, feature_vector=feature_vec)
                hog_features.append(features)
                hog_img[:,:,channel] = hog_image
            else:
                features = hog(img[:,:, channel], orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=True,visualise=False,
                    feature_vector=feature_vec)
                hog_features.append(features)
        hog_features = np.ravel(hog_features)
        if vis == True:
            return hog_features, hog_img
        else:
            return hog_features


    def bin_spatial(self, img, size=(32, 32)):
        features = cv2.resize(img, size).ravel()
        return features


    def color_hist(self, img, nbins=32):
        channel1_hist = np.histogram(img[:,:,0], bins=nbins)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins)
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features


    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        window_list = []
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                window_list.append(((startx, starty), (endx, endy)))
        return window_list


    def search_windows(self, img, windows, clf, scaler,
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    spatial_feat=True,
                    hist_feat=True, hog_feat=True):

        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = FeatureExtractor.get_features(test_img,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = clf.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows



    def add_heat(self, img, bbox_list):
        heatmap = np.zeros_like(img[:,:,0])
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        return heatmap



    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap



    def draw_labeled_bboxes(self, main_img, labels):
        img = main_img.copy()
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        return img



    def find_boxes(self, main_img, bbox_list, threshold=2):
        img = main_img.copy()
        heatmap = FeatureExtractor.apply_threshold(FeatureExtractor.add_heat(img.copy(), bbox_list),
                                                    threshold=threshold)
        labels = label(heatmap)
        return FeatureExtractor.draw_labeled_bboxes(img, labels)