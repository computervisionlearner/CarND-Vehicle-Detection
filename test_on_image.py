# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:17:09 2017

@author: yang
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import utils
import glob
from scipy.ndimage.measurements import label
from skimage.feature import hog

dist_pickle = pickle.load(open("train_dist.p", "rb"))
svc = dist_pickle["clf"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles=False):
    # array of rectangles where cars were detected
    windows = []

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img)

    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # select colorspace channel for HOG
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1
    nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = utils.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = utils.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = utils.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell



            test_prediction = svc.predict(hog_features)

            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                windows.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return windows


def search_car(img):
    draw_img = np.copy(img)
    # img = img.astype(np.float32) / 255
    
    # all_windows = []

    # X_start_stop = [[None, None], [None, None], [None, None], [None, None]]
    # w0, w1, w2, w3 = 64, 96, 128, 196
    # o0, o1, o2, o3 = 0.75, 0.75, 0.75, 0.75
    # XY_window = [(w0, w0), (w1, w1), (w2, w2), (w3, w3)]
    # XY_overlap = [(o0, o0), (o1, o1), (o2, o2), (o3, o3)]
    # yi0, yi1, yi2, yi3 = 400, 400, 400, 400
    # Y_start_stop = [[yi0, yi0 + w0 * 1.25], [yi1, yi1 + w1 * 1.25], [yi2, yi2 + w2 * 1.25], [yi3, yi3 + w3 * 1.25]]
    #
    #
    #
    # for i in range(len(Y_start_stop)):
    #     windows = utils.slide_window(img, x_start_stop=X_start_stop[i], y_start_stop=Y_start_stop[i],
    #                         xy_window=XY_window[i], xy_overlap=XY_overlap[i])
    #
    #     all_windows += windows

    # on_windows = utils.search_windows(img, all_windows, svc, X_scaler, spatial_feat=True, hist_feat=True,
    #                                   hog_channel='ALL')
    windows = []

    colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    ystart = 400
    ystop = 464
    scale = 1.0
    windows+=(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 416
    ystop = 480
    scale = 1.0
    windows+=(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 496
    scale = 1.5
    windows+=(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 528
    scale = 1.5
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 528
    scale = 2.0
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 560
    scale = 2.0
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 596
    scale = 3.5
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    ystart = 464
    ystop = 660
    scale = 3.5
    windows += (find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                          orient, pix_per_cell, cell_per_block, None, None))
    
#    window_list = utils.slide_window(img)

    
    heat_map = np.zeros(img.shape[:2])
    heat_map = utils.add_heat(heat_map,windows)
    heat_map_thresholded = utils.apply_threshold(heat_map,1)
    labels = label(heat_map_thresholded)
    draw_img = utils.draw_windows(draw_img,windows)
    
    
#    draw_img = utils.draw_windows(draw_img,on_windows)
    return draw_img


ystart = 400
ystop = 656
scale = 1.5

X_start_stop = [[None, None], [None, None], [None, None], [None, None]]
w0, w1, w2, w3 = 64, 96, 128, 196
o0, o1, o2, o3 = 0.75, 0.75, 0.75, 0.75
XY_window = [(w0, w0), (w1, w1), (w2, w2), (w3, w3)]
XY_overlap = [(o0, o0), (o1, o1), (o2, o2), (o3, o3)]
yi0, yi1, yi2, yi3 = 400, 400, 400, 400
Y_start_stop = [[yi0, yi0 + w0 * 1.25], [yi1, yi1 + w1 * 1.25], [yi2, yi2 + w2 * 1.25], [yi3, yi3 + w3 * 1.25]]


test_imgs = []
out_imgs = []
img_paths = glob.glob('test_images/*.jpg')
plt.figure(figsize=(20, 68))
for path in img_paths:
    img = mpimg.imread(path)
    
    out_img = search_car(img)
    test_imgs.append(img)
    out_imgs.append(out_img)

plt.figure(figsize=(20, 68))
for i in range(len(test_imgs)):
    plt.subplot(2 * len(test_imgs), 2, 2 * i + 1)
    #    plt.title('before thresholds')
    plt.imshow(test_imgs[i])

    plt.subplot(2 * len(test_imgs), 2, 2 * i + 2)
    #    plt.title('after thresholds')
    plt.imshow(out_imgs[i][0],cmap='gray')
    
    
    
#data = glob.glob('data_examples/*/*.png')
#data_imgs=[]
#hog_imgs = []
#for i in range(len(data)):
#    data_img = mpimg.imread(data[i])
#    features, hog_img = hog(data_img[:,:,2], orientations=9, pixels_per_cell=(8, 8),
#                       cells_per_block=(2, 2), transform_sqrt=False, 
#                       visualise=True)
#    data_imgs.append(data_img)
#    hog_imgs.append(hog_img)
#
#plt.figure(figsize=(20, 68))
#for i in range(len(data)):
#    plt.subplot(2 * len(data), 2, 2 * i + 1)
#    plt.title('origin')
#    plt.imshow(data_imgs[i])
#
#    plt.subplot(2 * len(data), 2, 2 * i + 2)
#    plt.title('hog')
#    plt.imshow(hog_imgs[i],cmap='gray')
#    