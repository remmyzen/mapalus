
import tensorflow as tf
from tensorflow import keras
import numpy as np

class EnergyOverlapCallback(keras.callbacks.Callback):
    def __init__ (self, hamiltonian, num_samples, y_data, stopping_threshold=0.0):
        super(EnergyOverlapCallback, self).__init__()
        self.y_data = y_data
        self.hamiltonian = hamiltonian
        self.num_samples = num_samples
        self.stopping_threshold = stopping_threshold
       
    def on_epoch_end(self, epoch, logs):
        num_visible = self.hamiltonian.graph.num_points

        num_conf = 2 ** num_visible
        confs = []
        for i in range(num_conf):
            conf_bin = format(i, '#0%db' % (num_visible + 2))
            ## configuration in binary -1 1
            conf = np.array([1 if c == '1' else -1 for c in conf_bin[2:]])
            confs.append(conf)
        confs = np.array(confs)

        wv = tf.exp(self.model.predict(confs, batch_size=8000))
        probs = wv ** 2 / tf.reduce_sum(wv ** 2)
    
        samples_index = np.random.choice(range(num_conf), self.num_samples, p=probs.numpy().flatten(), replace=True)        
        samples = confs[samples_index, :]
          
        hamil = self.hamiltonian.calculate_hamiltonian_matrix(samples, self.num_samples)
        lvd = self.hamiltonian.calculate_ratio(samples, self, self.num_samples)
        eloc_array = tf.reduce_sum((tf.exp( lvd) * hamil), axis=1, keepdims=True)

        wv = wv.numpy()

        overlap = np.abs(np.dot(wv.T, self.y_data)) ** 2 / np.sum(np.abs(wv) ** 2)

        logs['overlap'] = overlap
        logs['eloc_mean'] = np.mean(eloc_array)
        logs['eloc_std'] = np.std(eloc_array)
        
        if self.stopping_threshold > 0:
            print(' - overlap: %.3f - energy: %.3f - stopping: %.3f' % (overlap, np.mean(eloc_array), np.std(eloc_array)/ np.abs(np.mean(eloc_array))))
            if np.std(eloc_array) / np.abs(np.mean(eloc_array)) < self.stopping_threshold:
                self.model.stop_training = True
            

    def log_val(self, x):
        """
            Calculate log(\Psi(x)), model are trained to learn log(\Psi(x))
            Args:
                x: the x
        """
        return self.model(x, training=True)

    def log_val_diff(self, xprime, x):
        """
            Calculate log(\Psi(x')) - log(\Psi(x))
            Args:
                xprime: x'
                x: x
        """
        log_val_1 = self.log_val(xprime)
        log_val_2 = self.log_val(x)
        return log_val_1 - log_val_2

