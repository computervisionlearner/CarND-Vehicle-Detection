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

dist_pickle = pickle.load(open("train_dist.p", "rb"))
svc = dist_pickle["clf"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = utils.convert_color(img_tosearch, conv='RGB2YCrCb')
    print(ctrans_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = utils.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = utils.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = utils.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = utils.bin_spatial(subimg, size=spatial_size)
            hist_features = utils.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                hog_features.reshape(1, -1))
            #            test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


def search_car(img):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    all_windows = []

    # X_start_stop =[[None,None],[None,None],[None,None],[None,None]]
    # w0,w1,w2,w3 = 240,180,120,70
    # o0,o1,o2,o3 = 0.75,0.75,0.75,0.75
    # XY_window = [(w0,w0),(w1,w1),(w2,w2),(w3,w3)]
    # XY_overlap = [(o0,o0),(o1,o1),(o2,o2),(o3,o3)]
    # yi0,yi1,yi2,yi3 = 380,380,395,405
    # Y_start_stop =[[yi0,yi0+w0*1.25],[yi1,yi1+w1*1.25],[yi2,yi2+w2*1.25],[yi3,yi3+w3*1.25]]

    X_start_stop = [[None, None], [None, None], [None, None], [None, None]]
    w0, w1, w2, w3 = 64, 96, 128, 196
    o0, o1, o2, o3 = 0.75, 0.75, 0.75, 0.75
    XY_window = [(w0, w0), (w1, w1), (w2, w2), (w3, w3)]
    XY_overlap = [(o0, o0), (o1, o1), (o2, o2), (o3, o3)]
    yi0, yi1, yi2, yi3 = 400, 400, 400, 400
    Y_start_stop = [[yi0, yi0 + w0 * 1.25], [yi1, yi1 + w1 * 1.25], [yi2, yi2 + w2 * 1.25], [yi3, yi3 + w3 * 1.25]]

    for i in range(len(Y_start_stop)):
        windows = utils.slide_window(img, x_start_stop=X_start_stop[i], y_start_stop=Y_start_stop[i],
                                     xy_window=XY_window[i], xy_overlap=XY_overlap[i])

        all_windows += windows

    # window_list = utils.slide_window(img)
    on_windows = utils.search_windows(img, all_windows, svc, X_scaler, spatial_feat=True, hist_feat=True,
                                      hog_channel='ALL')

    heat_map = np.zeros(img.shape[:2])
    heat_map = utils.add_heat(heat_map, on_windows)
    heat_map_thresholded = utils.apply_threshold(heat_map, 1)
    labels = label(heat_map_thresholded)
    draw_img = utils.draw_labeled_bboxes(draw_img, labels)

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
img_paths = [img_paths[0]]
for path in img_paths:
    img = mpimg.imread(path)
    windows = utils.slide_window(img, x_start_stop=X_start_stop[3], y_start_stop=Y_start_stop[3],
                                 xy_window=XY_window[3], xy_overlap=XY_overlap[3])
    # #    out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
    # #                    hist_bins)
    # out_img = search_car(img)
    out_img = utils.draw_windows(img,windows)
    test_imgs.append(img)
    out_imgs.append(out_img)

plt.figure(figsize=(20, 68))
for i in range(len(test_imgs)):
    plt.subplot(2 * len(test_imgs), 2, 2 * i + 1)
    #    plt.title('before thresholds')
    plt.imshow(test_imgs[i])

    plt.subplot(2 * len(test_imgs), 2, 2 * i + 2)
    #    plt.title('after thresholds')
    plt.imshow(out_imgs[i])