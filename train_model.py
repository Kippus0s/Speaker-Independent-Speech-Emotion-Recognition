# train_model.py 
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import argparse
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AirAbsorption
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, Callback

from prepare_tf_datasets import *
from models import model_specs

# Argument handling
def parse_args():
    parser = argparse.ArgumentParser(description="Train SER model")

    parser.add_argument(
        "DATASET",
        type=str,
        choices=["emodb", "iemocap", "ravdess", "savee"],
        help="Dataset name"
    )

    parser.add_argument(
        "DATATYPE",
        type=str,
        choices=["wav", "mel", "mfcc"],
        help="Input data representation"
    )

    parser.add_argument(
        "SAMPLE_RATE",
        type=int,
        help="Audio sample rate"
    )

    parser.add_argument(
        "SAMPLE_DURATION",
        type=int,
        help="Audio duration in seconds"
    )

    parser.add_argument(
        "BATCH_SIZE",
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        "PREPROCESSED_ROOT_DIR",
        type=str,
        help="Preprocessed dataset subdirectory, this is the same as output directory used in dataset_preprocess.py"

    )

    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the model configuration to train, in the models subdirectory"

    )

    args = parser.parse_args()
    return args


args = parse_args()

def main(args):

    DATASET = args.DATASET
    DATATYPE = args.DATATYPE
    SAMPLE_RATE = args.SAMPLE_RATE
    SAMPLE_DURATION = args.SAMPLE_DURATION
    BATCH_SIZE = args.BATCH_SIZE
    PREPROCESSED_ROOT_DIR = args.PREPROCESSED_ROOT_DIR    
    model_name = args.model_name

    #Running utility functions to get the dataset path and shape of the data (duration, sample size and data representation all factors)
    DATASET_PATH = get_dataset_path(DATATYPE, PREPROCESSED_ROOT_DIR, DATASET)
    root_path = os.path.join(os.getcwd(), DATASET_MAP[DATASET.lower()])
    df_train = pd.read_csv(os.path.join(root_path, "train.csv"))
    INPUT_SHAPE = get_input_shape(DATATYPE, DATASET_PATH, df_train, SAMPLE_RATE, SAMPLE_DURATION)
  
    #Creating the tensorflow dataset objects, and building the model, passing the training data to normalise
    train_ds, val_ds, test_ds, INPUT_SHAPE = create_tf_datasets(DATASET, DATATYPE, SAMPLE_RATE, SAMPLE_DURATION, BATCH_SIZE, PREPROCESSED_ROOT_DIR)


    # turn the string into a callable
    model_name = getattr(model_specs, model_name)
    model, opt = model_name(INPUT_SHAPE, train_ds) 
    model.compile(optimizer = opt,loss='sparse_categorical_crossentropy',
                metrics=['accuracy']) 
    print("Model compiled successfully")

if __name__ == "__main__":
    main(args)

