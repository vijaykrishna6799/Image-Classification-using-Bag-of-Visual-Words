#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 03:14:06 2022

@author: vijaykrishna
"""


""" Reference from
https://github.com/bnsreenu/python_for_microscopists/blob/master/069a-Train_BOVW_V1.0.py
"""

"""
All images resized to 128 x 128
Images used for test are completely different that the ones used for training.
Used fruits dataset used in the assignment(FIDS 30 dataset)
4 images for testing, each of acerolas, guava, cantaloupe (4 x 3)
18 images for training, each of acerolas, guava, cantaloupe (18 x 3)
"""


import cv2
import numpy as np
import os

# extracting the training classes and storing them in a list
# Here we use folder names for class names 
#(so we get the names of the labels as that of the folder names)

# Names are acerola, guava, cantaloupe (selected 3 fruits from the FIDS30 dataset)
train_image_path = '/Users/vijaykrishna/ML mini/train'  
training_names = os.listdir(train_image_path)
# os.listdir() gives the list of all files in the path
# Get path to all images and save them in a list (Image_paths)
# for each image storing the class names of the image 
#(Image class names are given as the folder names using 'os' package)
image_paths = []
image_classes = []
class_id = 0

# we define a function to list all names 
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

# Fill the image_paths and image_classes and also add class ID number
    
for training_name in training_names:
    dir = os.path.join(train_image_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
# for each image we extract the key points using cv2 package and then store them in a numpy array
# Create feature extraction and keypoint detector objects
# Create List where all the descriptors will be stored
des_list = []

# BRISK is used to perform feature based image segmentation using open cv2 function
# ORB also might work
# key point features are retrieved by the function brisk.detectAndCompute and creating a list to store the values
brisk = cv2.BRISK_create(30)

for image_path in image_paths:
    im = cv2.imread(image_path) 
    kpts, des = brisk.detectAndCompute(im, None)
    des_list.append((image_path, des))   
# data generated by the detectAndCompute function is converted into a numpy array   
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

#kmeans works only on float, so converting integers to float
descriptors_float = descriptors.astype(float)  
# k-means clustering tries to group similar kind of items in the form of clusters 
#and also it finds the similarities between items and groups them into clusters
# collection of data points aggregated together because of similarities
# reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html
# Performing k-means clustering and vector quantization (vq)
from scipy.cluster.vq import kmeans, vq

k = 200  #k means with 100 clusters gives lower accuracy for the considered fruits datasets 
voc, variance = kmeans(descriptors_float, k, 1) 
# 'voc' vocabulary is nothing but the total features of the image
# Calculate the histogram of features and represent them as vector
#vq Assigns codes from a code book to observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
#Standardize features by removing the mean and scaling to unit variance
#In a way normalization
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

#Train an algorithm to discriminate vectors corresponding to positive and negative training images
# Train the Linear SVM
#from sklearn.svm import LinearSVC
#clf = LinearSVC(max_iter=1000)  #Default of 100 is not effective as the output is not converging
#clf.fit(im_features, np.array(image_classes))

#Train Random forest to compare how it does against SVM
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1000, random_state=30)
clf.fit(im_features, np.array(image_classes))


# Save the SVM and RFC to be used in the validation stage
#We use the joblib package and dump the objects into a pickle file
# pickle allows the objects to be serialized to files on disk 
# the pickle files generated can be deserialized back into the program while runtime
#Joblib dumps Python object into one file
import joblib
joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)    
    
# increasing the k-means clustering values increases the accuracy.

