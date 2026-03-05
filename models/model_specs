import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.layers import *

def emodb_wav (INPUT_SHAPE, train_ds):
    for x, y in train_ds.take(1):
        print(x.shape, y.shape)
    #Normalise on the entire trainingdataset (Rather than each utterance individually)
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))

    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv1D(16, kernel_size=5, activation='relu', padding='same')(x)  
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(32, kernel_size=5, activation='relu', padding='same')(x) 
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(48, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(96, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    #No batch norm here
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(160, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    output = Dense(7,activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    return model, opt

def next_models():
