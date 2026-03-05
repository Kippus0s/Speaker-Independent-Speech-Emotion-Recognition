# prepare_tf_datasets.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import argparse
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AirAbsorption

# ex. create_tf_datasets(emodb mfcc 16000 4 32 fixedduration)

# Setting up
"""Seeding and consistency via enable_op_determinism
This is to ensure parity between repeated tests"""
tf.keras.backend.clear_session()
tf.config.experimental.enable_op_determinism()
seed = 69
SEED = 69
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Setting up paths and loading the data splits
DATASET_MAP = {
        "emodb": "EmoDB",
        "iemocap": "IEMOCAP",
        "ravdess": "RAVDESS",
        "savee": "SAVEE"
    }

LABEL_MAP = {
     'emodb': {'anger': 0, 'boredom': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'neutral': 6},
     'iemocap': {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}, #Removed minority classes including "frustration" and combine happy and excited, this was done during creation of iemocap.csv
     'ravdess': {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7},
     'savee': {'anger': 0, 'disgust': 1, 'fear': 2, 'happiness': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6}
}


def get_dataset_path(DATATYPE, PREPROCESSED_ROOT_DIR, DATASET):
    # Adjust capitalisation to match original dataset directory naming
    dataset_name = DATASET_MAP[DATASET.lower()]

    # Match mfcc directory naming from dataset_preprocess.py
    if DATATYPE == "mfcc":
        datatype_temp = "mfccs"
    else:
        datatype_temp = DATATYPE.lower()

    # Construct dataset path
    if DATATYPE == "wav":
        DATASET_PATH = os.path.join(
            os.getcwd(),
            dataset_name,
            PREPROCESSED_ROOT_DIR
        )
    else:
        DATASET_PATH = os.path.join(
            os.getcwd(),
            dataset_name,
            PREPROCESSED_ROOT_DIR,
            datatype_temp
        )

    print("DATASET_PATH =", DATASET_PATH)
    return DATASET_PATH


def get_input_shape(DATATYPE, DATASET_PATH, df_train, SAMPLE_RATE, SAMPLE_DURATION):
    #Offline dataset creation with uniform samples, inspect first to determine data shape without having to refer to mel/mfcc cr eation constants and calculate
    first_file = df_train.iloc[0]['file']
    if DATATYPE in ("mfcc", "mel"):
        path = os.path.join(DATASET_PATH, os.path.normpath(first_file[:-4] + ".npy"))
        sample = np.load(path)
        INPUT_SHAPE = sample.shape + (1,)   # add channel dimension

    else:   
        #WAV 
        INPUT_SHAPE = (SAMPLE_DURATION * SAMPLE_RATE, 1)
    return INPUT_SHAPE 


# Defining the audio augmentations for use by Audiomentations library

augment = Compose([
    AirAbsorption(min_distance=10.0,max_distance=50.0,p=1.0),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)])

def load_augment(path):
    data, samplerate = sf.read(path) # returns tf tensor array    
    augmented_data = augment(samples=data,sample_rate=samplerate)    
    return tf.convert_to_tensor(augmented_data, dtype=tf.float32) # this is one file , i can pass this instead

def load_wav(path):    
    data, samplerate = sf.read(path) # returns tf tensor array    
    print(path)
    return tf.convert_to_tensor(data, dtype=tf.float32) 


# A modified process_file function which implements audio augmentations replace this with condition check for process file
def process_file_aug(file, DATASET, DATASET_PATH, INPUT_SHAPE):
    path = os.path.join(DATASET_PATH, file['file'])
    audio_tensor = load_augment(path)
    audio_tensor = tf.reshape(audio_tensor, INPUT_SHAPE)
    label = LABEL_MAP[DATASET][file['emotion']]
    return audio_tensor, label

def process_file(file, DATASET, DATATYPE, DATASET_PATH, INPUT_SHAPE):
        
        if DATATYPE == 'mfcc' or DATATYPE == 'mel':
            path = os.path.join(DATASET_PATH,os.path.normpath(file['file'][:-4]+".npy"))
            data = np.load(path)
            #data_shape = data.shape
            data = tf.reshape(data, INPUT_SHAPE)
        
        else: 
            if DATASET == "savee":
                path = os.path.join(DATASET_PATH, os.path.normpath(file['file']) + ".wav") 
            else: 
                path = os.path.join(DATASET_PATH, os.path.normpath(file['file']))
            data = load_wav(path)
            data = tf.reshape(data, INPUT_SHAPE)        
        label = LABEL_MAP[DATASET][file['emotion']]
        return data, label

def create_tf_dataset(df, DATASET, DATATYPE, DATASET_PATH, BATCH_SIZE, INPUT_SHAPE, shuffle=True):
    bs = BATCH_SIZE
    audio_label_pairs = [process_file(file, DATASET, DATATYPE, DATASET_PATH, INPUT_SHAPE) for _, file in df.iterrows()]# file 
    ds = tf.data.Dataset.from_generator(
    lambda: iter(audio_label_pairs),
    output_signature=(
        tf.TensorSpec(shape= INPUT_SHAPE, dtype=tf.float32),  # audio 
        tf.TensorSpec(shape=(), dtype=tf.int32)          # label
    ))
   
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df),seed=seed)
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def create_tf_dataset_aug(df, DATASET, DATASET_PATH, INPUT_SHAPE, BATCH_SIZE, shuffle=True):
    bs = BATCH_SIZE
    audio_label_pairs = [process_file_aug(file, DATASET, DATASET_PATH, INPUT_SHAPE) for _, file in df.iterrows()]# file 
    ds = tf.data.Dataset.from_generator(
    lambda: iter(audio_label_pairs),
    output_signature=(
        tf.TensorSpec(shape=(INPUT_SHAPE), dtype=tf.float32),  # audio
        tf.TensorSpec(shape=(), dtype=tf.int32)          # label
    ))
    #ds = ds.map(tf_load_wav, num_parallel_calls=tf.data.AUTOTUNE) #further preprocessing 
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df),seed=seed)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Create the datasets using tf dataset function

def create_datasets(df_train, df_val, df_test, DATASET, DATATYPE, SAMPLE_RATE, SAMPLE_DURATION, BATCH_SIZE, DATASET_PATH, INPUT_SHAPE):
    train_ds = create_tf_dataset(df_train, DATASET, DATATYPE, DATASET_PATH, BATCH_SIZE, INPUT_SHAPE)
    if DATATYPE == "wav":
        train_ds_aug = create_tf_dataset_aug(df_train, DATASET, DATASET_PATH, INPUT_SHAPE, BATCH_SIZE)
        train_ds = train_ds.concatenate(train_ds_aug)
    val_ds   = create_tf_dataset(df_val, DATASET, DATATYPE, DATASET_PATH, BATCH_SIZE, INPUT_SHAPE, shuffle=False)
    test_ds  = create_tf_dataset(df_test, DATASET, DATATYPE, DATASET_PATH, BATCH_SIZE, INPUT_SHAPE, shuffle=False)
    return train_ds, val_ds, test_ds

def create_tf_datasets(DATASET, DATATYPE, SAMPLE_RATE, SAMPLE_DURATION, BATCH_SIZE, PREPROCESSED_ROOT_DIR):

    #Creating the tensorflow dataset objects
    root_path = os.path.join(os.getcwd(), DATASET_MAP[DATASET.lower()])
    df_train = pd.read_csv(os.path.join(root_path, "train.csv"))
    df_val   = pd.read_csv(os.path.join(root_path, "val.csv"))
    df_test  = pd.read_csv(os.path.join(root_path, "test.csv"))

    DATASET_PATH = get_dataset_path(DATATYPE, PREPROCESSED_ROOT_DIR, DATASET)
    INPUT_SHAPE = get_input_shape(DATATYPE, DATASET_PATH, df_train, SAMPLE_RATE, SAMPLE_DURATION)
       

    train_ds, val_ds, test_ds =  create_datasets(df_train, df_val, df_test, DATASET, DATATYPE, SAMPLE_RATE, SAMPLE_DURATION, BATCH_SIZE, DATASET_PATH, INPUT_SHAPE)
       
    
    #Testing the dataset objects are created properly. 
    for audio, label in train_ds.unbatch().take(1):
        print("Audio shape:", audio.shape)
        print("Label:", label.numpy())
        print("Audio snippet:", audio[:10].numpy())

    return train_ds, val_ds, test_ds    


