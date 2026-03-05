import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
def class_weight_maker(ds):
    ds_unbatched = tuple(ds.unbatch()) #The datasets made are always batched and need to be unbatched to count the number of classes correctly
    TRAIN_DS_SIZE =  len(ds_unbatched)
    """Using method from "Class weights Calculate class weights" section in https://www.tensorflow.org/tutorials/structured_data/imbalanced_data """
    #class_weight parameter in fit() requires a dictionary, so I iteratively fill it using the formula from above link, and the counts collection that was provided in the sample dataset
    #Creating the class_weight dictionary
    print("train ds size = " ,TRAIN_DS_SIZE) # All of the print statements are to help me finish writing the function, to facilitate debugging mid-execution on test runs.
    labels = []
    for (item,label) in ds_unbatched:
        labels.append(label.numpy())
    labels = pd.Series(labels)
    print(labels) # This shows all items by index, and the value is the emotion label for a given item
    counts = labels.value_counts().sort_index() # Counts variable contains count of all classes in the dataset
    print(counts)
    class_weight = {}
    x = 0
    for i in counts.values: #Iterate through the values in counts        
        print("len counts num classes",len(counts)) 
        y =  (1 / i) * TRAIN_DS_SIZE /len(counts) #Incase of discrepancy between this
        print("y = ", y)
        class_weight.update({x: y}) 
        x=x+1 #Incremement in loop variable to move on to setting count of the next class.
    return class_weight
    
