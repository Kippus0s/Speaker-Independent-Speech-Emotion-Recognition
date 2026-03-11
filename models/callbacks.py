import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
import time 

class EpochTimer(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds.")


        

class Earlystop_1(EarlyStopping):
    def __init__(self): 
        super().__init__(monitor='val_loss', 
                        patience=3,restore_best_weights=True)          
                               

class Earlystop_2(EarlyStopping):
    def __init__(self):
        super().__init__(monitor='val_loss',  
                               patience=5, start_from_epoch = 20,         
                               restore_best_weights=True)        

class Earlystop_3(EarlyStopping):
    def __init__(self):
        super().__init__(monitor='val_loss',  
                               patience=10, start_from_epoch = 10,         
                               restore_best_weights=True)        
        

class Earlystop_4(EarlyStopping):
    def __init__(self):
        super().__init__(monitor='val_loss',  
                        patience=5,           
                        restore_best_weights=True,
                        )        
        
class Earlystop_5(EarlyStopping):
    def __init__(self):
        super().__init__(monitor='val_loss',  
                        patience=6,  start_from_epoch = 15,             
                        restore_best_weights=True,
                        )        
                                        
                                


class Plateau_decay_1(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self):
        super().__init__(
                        monitor='val_accuracy',
                        factor=0.1,
                        patience=3,
                        verbose=0,
                        mode='auto',
                        min_delta=0.0001,
                        cooldown=0,
                        min_lr=1e-6
                        )
        
class Plateau_decay_2(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self):
        super().__init__(monitor='val_accuracy',
    factor=0.1,
    patience=10,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-6,
)

class Plateau_decay_3(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self):
        super().__init__(monitor='val_accuracy',
    factor=0.85,
    patience=3,
    verbose=0,
    start_from_epoch=3,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-6,
)



