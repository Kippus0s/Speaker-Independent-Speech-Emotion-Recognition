# train_model.py 
import os
import sys
import time
import json
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

from evaluate import *
from prepare_tf_datasets import *

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "utilities"))
sys.path.append(os.path.join(root_dir, "models"))
from class_weight import class_weight_maker
import model_specs, callbacks

"""
Arguments:
Ex: emodb mfcc 16000 4 16 norm_and_fixedduration emodb_mfcc y

DATASET
Choices: "emodb", "iemocap", "ravdess", "savee"
Specifies which dataset to use for training. Used for dataset path resolution and loading appropriate CSV metadata.

DATATYPE
Choices: "wav", "mel", "mfcc"
Defines the type of input representation (raw waveform, Mel spectrogram, or MFCC features). Determines preprocessing and model input shape.

SAMPLE_RATE
Sets the audio sampling rate for loading or interpreting audio files. Used in dataset creation to ensure consistent temporal resolution.

SAMPLE_DURATION
Sets the length of audio clips in seconds. Influences input tensor shape and dataset preparation.

BATCH_SIZE
Defines the batch size for training and validation. Impacts memory usage and gradient update frequency.

PREPROCESSED_ROOT_DIR
Directory where preprocessed data is stored. Optionally set in dataset_preprocess arguments, or "fixedduration"or "norm_and_fixedduration" if left to default depending on if z_score normalisation was selected in dataset_preprocess arguments
Used to locate the preprocessed dataset files

model_name
Specifies the model configuration to use from models.model_specs. Dynamically converted to a callable that returns a compiled model.

normalise_class_weights
Choices: "y", "n"
Default: "n"
Purpose: If "y", class weights are normalized to balance the contribution of each class during training. Affects the class_weight argument in model.fit().

--output
optional
filename suffix, for example, entering TEST will result in a filename of dataset_datatype_TEST_weights

"""


#Determinism via seeding and enforcing deterministic gpu operations - ensuring identical results between runs
tf.keras.backend.clear_session()
tf.config.experimental.enable_op_determinism()
seed = 69
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

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

    parser.add_argument(
        "normalise_class_weights",
        type=str,
        default="n",
        choices=["y", "n"],
        help="Normalise class weights, makes a moderate difference in most cases for better or for worse"
    )
  
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional suffix to append to the the filename, so predictions and weights do not overwrite previous experiments," \
        "for example, entering TEST will result in a filename of dataset_datatype_TEST_weights"
        )    
    
    args = parser.parse_args()
    return args


def main(args):
    #Take args
    DATASET = args.DATASET
    DATATYPE = args.DATATYPE
    SAMPLE_RATE = args.SAMPLE_RATE
    SAMPLE_DURATION = args.SAMPLE_DURATION
    BATCH_SIZE = args.BATCH_SIZE
    PREPROCESSED_ROOT_DIR = args.PREPROCESSED_ROOT_DIR    
    filename_suffix = args.output
    model_name = args.model_name
    cw_flag = args.normalise_class_weights
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #Running utility functions to get the dataset path and shape of the data (duration, sample size and data representation all factors)
    DATASET_PATH = get_dataset_path(DATATYPE, PREPROCESSED_ROOT_DIR, DATASET)
    dataset_root = os.path.join(root_dir, "datasets", DATASET_MAP[DATASET.lower()])
    df_train = pd.read_csv(os.path.join(dataset_root, "train.csv"))
    INPUT_SHAPE = get_input_shape(DATASET, DATATYPE, DATASET_PATH, df_train, SAMPLE_RATE, SAMPLE_DURATION)
  
    #Creating the tensorflow dataset objects, and building the model, passing the training data to normalise
    train_ds, val_ds, test_ds, INPUT_SHAPE = create_tf_datasets(DATASET, DATATYPE, SAMPLE_RATE, SAMPLE_DURATION, BATCH_SIZE, PREPROCESSED_ROOT_DIR)

    # Data integrity check 
    for spect, label in train_ds.unbatch().take(3):
        print("Data shape:", spect.shape)
        print("Label:", label.numpy())
        print("Snippet:", spect.numpy())
    print("datasets created succesfully")

    # Turn the model argument string into a callable to retrive spec from model_specs
    model_name = getattr(model_specs, model_name)
    #Compile model
    model, model_callbacks, epoch_count = model_name(INPUT_SHAPE, train_ds) 
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer = opt,loss='sparse_categorical_crossentropy',
                metrics=['accuracy']) 
    print("Model compiled successfully")

    #Train model
    if cw_flag:
        cw = class_weight_maker(train_ds)
    else: cw = None 

    history = model.fit(train_ds, 
                 epochs=epoch_count,
                 validation_data=val_ds,callbacks=[model_callbacks], class_weight = cw)
    
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Evaluation complete: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    # Saving training weights, history, plotting confusion matrix and saving the image, 
    # and rendering a class-by-class accuracy report
    filename = str(DATASET) + "_" + str(DATATYPE)
    if args.filename is not None:
        filename = str(DATASET) + "_" + str(DATATYPE) + filename_suffix 
    save_weight_history(model,history,filename,test_ds)
    save_preds_and_true(model, filename, test_ds)
    plot_cm(model, filename, test_ds, DATASET, LABEL_MAP)
    display_class_report(model,test_ds)
    
    
if __name__ == "__main__":
    #Derive cmdline args
    args = parse_args()
    main(args)

