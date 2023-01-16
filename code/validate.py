#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 03:17:31 2022

@author: vijaykrishna
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
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score 
import joblib


# Load the classifier, class names, scaler, number of clusters and vocabulary 
#from stored pickle file (generated during training phase)
# we unpack the contents of the pkl file
clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

# Get the path of the testing image(s) and store them in a list
#test_path = 'dataset/test' # Names are Aeroplane, Bicycle, Car
test_path = '/Users/vijaykrishna/ML mini/test'  # Folder Names are of fruits (acerolas, guava, cantaloupe)
#instead of test if you use train then we get great accuracy

testing_names = os.listdir(test_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function

def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the empty lists with image path, classes, and add class ID number
# class ID of each class will be names of the folders
# the code is the same but we only take different images for predictions
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
    
# Create feature extraction and keypoint detector objects
    #SIFT is not available anymore in openCV    
# Create List where all the descriptors will be stored
des_list = []

#BRISK is a good replacement to SIFT. ORB also works but didn;t work well for this example
brisk = cv2.BRISK_create(30)

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts, des = brisk.detectAndCompute(im, None)
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# Calculate the histogram of features
#vq Assigns codes from a code book to observations.
from scipy.cluster.vq import vq    
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
#Standardize features by removing the mean and scaling to unit variance
#Scaler (stdSlr comes from the pickle file we imported at the start of the code)
test_features = stdSlr.transform(test_features)

# Until here most of the above code is similar to train phase except for kmeans clustering

#true class names are considered so they can be compared with predicted classes
true_class =  [classes_names[i] for i in image_classes]
# Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf.predict(test_features)]


#Print the true class and Predictions 
print ("true_class ="  + str(true_class))
print ("prediction ="  + str(predictions))

#To make it easy to understand the accuracy let us print the confusion matrix

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


accuracy = accuracy_score(true_class, predictions)
print ("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print (cm)

showconfusionmatrix(cm)
#print('Confusion Matrix for SVC 1000 iterations')
print('Confusion Matrix for RF classifier for n-estimators=100')





