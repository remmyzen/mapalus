import time
import tensorflow
from tensorflow import keras

class TimeCallback(keras.callbacks.Callback):
 
    def __init__ (self):
        super(TimeCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs):
        logs['time'] = time.time() - self.epoch_time_start
    

