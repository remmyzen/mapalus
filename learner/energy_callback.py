
import tensorflow as tf
from tensorflow import keras
import numpy as np

class EnergyCallback(keras.callbacks.Callback):
    def __init__ (self, hamiltonian, sampler, stopping_threshold=0.0):
        super(EnergyCallback, self).__init__()
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.stopping_threshold = stopping_threshold
       
    def on_epoch_end(self, epoch, logs):
        num_visible = self.hamiltonian.graph.num_points
 
        if epoch == 0:
            self.samples = tf.convert_to_tensor(self.sampler.get_initial_random_samples(num_visible))

        samples = self.sampler.sample(self, self.samples, self.sampler.num_samples)
          
        hamil = self.hamiltonian.calculate_hamiltonian_matrix(samples, self.sampler.num_samples)
        lvd = self.hamiltonian.calculate_ratio(samples, self, self.sampler.num_samples)
        eloc_array = tf.reduce_sum((tf.exp( lvd) * hamil), axis=1, keepdims=True)

        self.samples = samples

        logs['eloc_mean'] = np.mean(eloc_array)
        logs['eloc_std'] = np.std(eloc_array)
        
        #print(' - energy: %.3f - stopping: %.3f' % (np.mean(eloc_array), np.std(eloc_array)/ np.abs(np.mean(eloc_array))), end=' ')
        print(' - energy: %.3f - stopping: %.3f' % (np.mean(eloc_array), np.std(eloc_array)/ np.abs(np.mean(eloc_array))))
        if self.stopping_threshold > 0:
            if np.std(eloc_array) / np.abs(np.mean(eloc_array)) < self.stopping_threshold:
                self.model.stop_training = True
            
    def is_real(self):
        return False

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


