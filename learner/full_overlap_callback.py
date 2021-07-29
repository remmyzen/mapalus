
import tensorflow as tf
from tensorflow import keras
import numpy as np

class FullOverlapCallback(keras.callbacks.Callback):
    def __init__ (self, num_visible, num_samples, y_data, stopping_threshold=0.0):
        super(FullOverlapCallback, self).__init__()
        self.y_data = y_data
        self.num_visible = num_visible
        self.num_samples = num_samples
        self.stopping_threshold = stopping_threshold
       
    def on_epoch_end(self, epoch, logs):

        ## Create all configurations
        num_conf = 2 ** self.num_visible
        confs = []
        for i in range(num_conf):
            conf_bin = format(i, '#0%db' % (self.num_visible + 2))
            ## configuration in binary -1 1
            conf = np.array([1 if c == '1' else -1 for c in conf_bin[2:]])
            confs.append(conf)
        confs = np.array(confs)

        ## Predict the wave function
        wv = tf.exp(self.model.predict(confs, batch_size=8000)).numpy()

        overlap = np.abs(np.dot(wv.T, self.y_data)) ** 2 / np.sum(np.abs(wv) ** 2)

        logs['overlap'] = overlap
        
        print(' - full overlap: %.3f' % (overlap), end =' ')
        if self.stopping_threshold > 0:
            if overlap > self.stopping_threshold:
                self.model.stop_training = True
