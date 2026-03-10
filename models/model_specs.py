import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.layers import *
from models import callbacks 

def emodb_wav (INPUT_SHAPE, train_ds):
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
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_2(), callbacks.Plataeu_decay_1()]
    epoch_count = 50
    return model, model_callbacks, epoch_count

def emodb_mel(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(32, kernel_size=(11,11))(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(7,7))(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, kernel_size=(5,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, kernel_size=(5,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x) # JUST ADDED THIS TRYINGTO BEAT 68.9
    output = Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_2(), callbacks.Plataeu_decay_2()]
    return model, model_callbacks, epoch_count

def emodb_mfcc(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))

    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(64, kernel_size=(11,11),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(7,7),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, kernel_size=(7,7),padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = GlobalAveragePooling2D()(x)
    x= Dropout(0.4)(x)
    x = Flatten()(x)
    output = Dense(7, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_3(), callbacks.Plataeu_decay_2()]
    return model, model_callbacks, epoch_count

def ravdess_wav(INPUT_SHAPE, train_ds):
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
    output = Dense(8, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 50
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_1(), callbacks.Plataeu_decay_1()]
    return model, model_callbacks, epoch_count

def ravdess_mel(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(32, kernel_size=(2,4),padding='same')(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,5))(x)
    x = Conv2D(64, kernel_size=(2,3),padding='same')(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,4))(x)
    x = Conv2D(128, kernel_size=(2,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,4))(x)
    x = Conv2D(256, kernel_size=(4,4), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x) 
    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(8, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_2(), callbacks.Plataeu_decay_2()]
    return (model,model_callbacks,epoch_count)

def ravdess_mfcc(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(64, kernel_size=(11,11),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(7,7),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, kernel_size=(7,7),padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = GlobalAveragePooling2D()(x)
    x= Dropout(0.4)(x)
    x = Flatten()(x)
    output = Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_3(), callbacks.Plataeu_decay_2()]
    return model, model_callbacks, epoch_count

def iemocap_wav(INPUT_SHAPE, train_ds):
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
    output = Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 50
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_1(), callbacks.Plataeu_decay_1()]
    return model, model_callbacks, epoch_count

def iemocap_mel(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(32, kernel_size=(3,3),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(4,4))(x)
    x = Conv2D(64, kernel_size=(2,2), padding='same')(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,3))(x)
    x = Conv2D(128, kernel_size=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,3))(x)
    x = Conv2D(256, kernel_size=(1,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_4(), callbacks.Plataeu_decay_1()]
    return model, model_callbacks, epoch_count

def iemocap_mfcc(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(32, kernel_size=(3,3),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(4,4))(x)
    x = Conv2D(64, kernel_size=(2,2), padding='same')(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,3))(x)
    x = Conv2D(128, kernel_size=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,3))(x)
    x = Conv2D(256, kernel_size=(1,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    output = Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100 
    model_callbacks = [callbacks.EpochTimer(), callbacks.Earlystop_5(), callbacks.Plataeu_decay_1()]
    return model, model_callbacks

def savee_wav(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)    
    norm_layer.adapt(train_ds.map(lambda x, y: x)) #test
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv1D(16, kernel_size=5, padding='same')(x)  
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(32, kernel_size=5,  padding='same')(x) 
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Conv1D(48, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(64, kernel_size=3,  padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(96, kernel_size=3,  padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(160, kernel_size=3,  padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = Conv1D(256, kernel_size=3,  padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    output = Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model_callbacks = [callbacks.EpochTimer()]
    epoch_count = 20
    return model, model_callbacks, epoch_count

def savee_mel(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(32, kernel_size=(11,11))(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(7,7))(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, kernel_size=(5,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, kernel_size=(5,5), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 100
    model_callbacks = [callbacks.EpochTimer()]
    return model, model_callbacks, epoch_count

def savee_mfcc(INPUT_SHAPE, train_ds):
    norm_layer = layers.Normalization(input_shape = INPUT_SHAPE)
    norm_layer.adapt(train_ds.map(lambda x, y: x))
    inputs = Input(INPUT_SHAPE)
    x = norm_layer(inputs)
    x = Conv2D(64, kernel_size=(11,11),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=(7,7),padding="same")(x)   
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128, kernel_size=(7,7),padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    epoch_count = 46
    model_callbacks = [callbacks.EpochTimer()]
    return model, model_callbacks, epoch_count
