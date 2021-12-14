#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import sklearn.metrics
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style("white")



def img_data_generator(img_path, img_height,img_width, BATCH_SIZE=64, test_ratio=0.2):
    """
    This is a customized function for generating training and test image data from 
    the built-in Keras module with pre-defined parameters. 
    
    Parameters
    -----------------
    img_path = path to image file directory 
    img_height,img_width = input image dimensions
    BATCH_SIZE = batch size of the model
    test_ratio = train-test split split
    
    
    Returns
    ------------------
    train_generator = keras tensor image object for training
    validation_generator = keras tensor image object for testing
    
    Notes
    ------------------
    The model augmentation are fixed here that can be changed from the internal
    fucntions if needed. Ther purpose is to use same settings across different
    models for comparison.
    
    
    References
    -----------------
    More information about Keras image data generator:
    https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image

    
    """
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    validation_split=test_ratio) 

    train_generator = train_datagen.flow_from_directory(
    img_path,
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale",
    subset='training') 

    validation_generator = train_datagen.flow_from_directory(
    img_path, 
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale",
    subset='validation') 
    
    return train_generator, validation_generator



def CNN_model(img_height, img_width, OPTIMIZER):
     """
    This is a customized function for generating a Keras model built-in Keras module 
    with pre-defined parameters and model architecture. 
    
    Parameters
    -----------------
    img_height,img_width = input image dimensions
    OPTIMIZER = keras optimizer to be used for the model. More info here:
    https://keras.io/api/optimizers/
    
    Returns
    ------------------
    model = the model that can be used to build, train, fit and evaluate on data
    
    Notes
    ------------------
    The model architecture is fixed here that can be changed from the internal
    fucntions if needed. Ther purpose is to use same settings across different
    models for comparison.
    
    
    References
    -----------------
    More information about Keras models:
    https://www.tensorflow.org/api_docs/python/tf/keras/Model

    """
    
    model = Sequential([
        Conv2D(16, 1, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=OPTIMIZER,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    
    return model



def CNN_model_basic(img_height, img_width,OPTIMIZER):
    """
    This is a customized function for generating a Keras model built-in Keras module 
    with pre-defined parameters and model architecture. 
    
    Parameters
    -----------------
    img_height,img_width = input image dimensions
    OPTIMIZER = keras optimizer to be used for the model. More info here:
    https://keras.io/api/optimizers/
    
    Returns
    ------------------
    model = the model that can be used to build, train, fit and evaluate on data
    
    Notes
    ------------------
    The model architecture is fixed here that can be changed from the internal
    fucntions if needed. Ther purpose is to use same settings across different
    models for comparison.
    
    
    References
    -----------------
    More information about Keras models:
    https://www.tensorflow.org/api_docs/python/tf/keras/Model

    
    """
    model_basic = Sequential([
        Conv2D(16, 1, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D(),
        Conv2D(64, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.8),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model_basic.compile(optimizer=OPTIMIZER,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    
    return model_basic



def model_metrics(model, test, history):
     """
    The function evaluates the model metrics and prints out test loss 
    and accuracy. It also stores the history generated by the model 
    in a text file.
    
    Parameters
    -----------------
    model = the model to be evaluated on
    test = the test samples as keras tensor object
    history = history file from the model fitting
    
    Returns
    ------------------
    score = python list of model metrics with loss, accuracy, precision and 
    recall respectively
    It also prints out the test loss and accuracy in four and two significant 
    figures.
    
    Notes
    ------------------
    The print function can be changed to what is needed for model comparison, 
    purpose is to use same settings across different models.
    
    
    References
    -----------------
    More information about Keras model metrics:
    https://www.tensorflow.org/api_docs/python/tf/keras/metrics

    """
    y_pred = (model.predict(test) > 0.5).astype(int)
    y_true = test.classes
    
    print("The model performance metrics are as follows:")
    score = model.evaluate(test, verbose=1)
    print('Test set loss: {0:.4f} and accuracy: {1:.2f}%'.format(score[0], score[1] * 100))
    #Saving the information as txt file
    with open(str(history) + '.txt', 'a+') as f:
        print(history.history, file=f)
        print(score, file=f)
    
    return score



def confusion_matrix(model, test):
    """
    The function calcualtes the model metrics from scikit-learn library
    and prints out F1 score and confusion matrix as a seaborn heatmap.
    It also print out the confusion matrix as a Pandas Dataframe.
    
    Parameters
    -----------------
    model = the model to be evaluated on
    test = the test samples as keras tensor object
    
    Returns
    ------------------
    conf_mat = confusion matrix of the model as Pandas Dataframe
    It also prints out the F1 score and confusion matrix as heatmap
    
    Notes
    ------------------
    The print function can be changed to what is needed for model comparison, 
    purpose is to use same settings across different models.
    
    
    References
    -----------------
    More information about scikit-learn model metrics:
    https://scikit-learn.org/0.16/modules/generated/
    sklearn.metrics.confusion_matrix.html

    """
    y_pred = (model.predict(test) > 0.5).astype(int)
    y_true = test.classes
    print(f'F1 score: {sklearn.metrics.f1_score(y_true, y_pred)}')
    conf_mat = pd.DataFrame(sklearn.metrics.confusion_matrix(y_true, y_pred), 
             columns=['Predicted Negative', 'Predicted Positive'], 
             index=['True Negative', 'True Positive'])
    plt.figure(figsize=(4,4))
    sns.set(font_scale=1)
    sns.heatmap(conf_mat, annot=True, linewidths=.5, fmt="d", cbar=False, cmap="Dark2")
    plt.title("Confusion matrix")
    plt.show()
    
    return conf_mat


# In[ ]:




