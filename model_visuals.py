#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os 
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import sklearn.metrics


def data_visualization(img_dir, samples=5, size=25):
    """
    The function takes in the path to image directory, along with number of samples
    figure size to be displayed the image as input.
    
    Parameters
    -----------------
    img_dir = path to image files
    samples = number of images to be displayed
    size = matlplotlib figure size
    
    
    Returns
    ---------------
    An image grid with specified number of samples and size.
    
    
    Notes
    --------------
    The function is customized to show images in grayscale and labels them according to
    its directory name. It uses libraries not commonly used and needs to be installed 
    for it to work properly.
    
    """
    path = os.path.join(img_dir,'*g') 
    all_images = glob.glob(path)
    selected_images = random.sample(all_images, samples)
    images = [] 
    
    for i in selected_images: 
        img = cv2.imread(i) 
        images.append(img)
        
    plt.figure(figsize=(size, size))
    for i in range(len(images)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        plt.title(os.path.basename(img_dir), color='black', fontsize=14)
        plt.axis('off')
        
        
        
def plot_acc_loss(history):
    """
    The function plots accuracy and loss curves from the history
    file saved in the model output
    
    Parameters
    --------------
    history = history file from model output 
    
    
    Returns
    --------------
    A matplotlib subplot of model accuracy and loss side by side
    for both training and test data
    
    
    Notes
    ---------------
    The fucntion has default settings of image size and styles that can be
    changed internally if needed. Purpose was to maintain consistency among
    different model evaluations.
    
    """
    
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    sns.set_style("white")
    ax[0].set_title('Accuracy')
    ax[0].plot(history.history['accuracy'], color = 'cornflowerblue')
    ax[0].plot(history.history['val_accuracy'], color = 'crimson')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Test'], loc='best')

    ax[1].set_title('Loss')
    ax[1].plot(history.history['loss'], color = 'cornflowerblue')
    ax[1].plot(history.history['val_loss'], color = 'crimson')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='best')
    plt.subplots_adjust(wspace=0.4)
    plt.show()
    
    
    
def plot_rec_prec(history):
    
    """
    The function plots recall and precision curves from the history
    file saved in the model output
    
    Parameters
    --------------
    history = history file from model output 
    
    
    Returns
    --------------
    A matplotlib subplot of model precision and recall side by side
    for both training and test data
    
    
    Notes
    ---------------
    The fucntion has default settings of image size and styles that can be
    changed internally if needed. Purpose was to maintain consistency among
    different model evaluations.
    
    """
    
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    sns.set_style("white")
    ax[0].set_title('Recall')
    ax[0].plot(history.history['recall'], color = 'cornflowerblue')
    ax[0].plot(history.history['val_recall'], color = 'crimson')
    ax[0].set_ylabel('Recall')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Test'], loc='best')

    ax[1].set_title('Precision')
    ax[1].plot(history.history['precision'], color = 'cornflowerblue')
    ax[1].plot(history.history['val_precision'], color = 'crimson')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='best')
    plt.subplots_adjust(wspace=0.4)
    plt.show()
    


# In[ ]:




