# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:55:37 2017

@author: yang
"""
import os
import utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib
import pickle
from sklearn.grid_search import GridSearchCV
# Divide up into cars and notcars


notcars = glob.glob('../non-vehicles/*/*.png')
cars = glob.glob('../vehicles/*/*.png')



# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size=(32,32)
hist_bins=32

t=time.time()
car_features = utils.extract_features(cars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,spatial_size=spatial_size,hist_bins=hist_bins)
notcar_features = utils.extract_features(notcars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel,spatial_size=spatial_size,hist_bins=hist_bins)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features))
X = X.astype(np.float64)                       
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)


print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1, 3, 5,8, 10]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
# Check the training time for the SVC
t=time.time()
clf.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train classfier...')
# Check the score of the SVC
print('Test Accuracy of classfier = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My classfier predicts: ', clf.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with classfier')


train_dist={}
train_dist['clf']=clf
train_dist['scaler']=X_scaler
train_dist['orient']=orient
train_dist['pix_per_cell'] = pix_per_cell
train_dist['cell_per_block'] = cell_per_block
train_dist['hog_channel'] = hog_channel
train_dist['spatial_size'] = spatial_size
train_dist['hist_bins'] = hist_bins

output = open('train_dist.p', 'wb')
pickle.dump(train_dist,output)

