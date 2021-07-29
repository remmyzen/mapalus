from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
import copy
import scipy.stats


class LearnerSupervised:
    """
    This class is used to specified all the learning process and saving data for logging purposes. 

    TODO: Minibatch training
    """

    def __init__(self, hamiltonian, model, reference_energy=None, 
                 observables=[], num_samples=1000):
        """
        Construct a learner objects
        Args:
            hamiltonian: Hamiltonian of the model
            model: the machine learning model used
            sampler: the sampler used to train
            optimizer: the optimizer for training
            num_epochs: the number of epochs for training (Default: 1000)
            minibatch_size: the number of minibatch training (Default: 0)
            window_period: the number of windows for logging purposes (Default: 50) 
            reference_energy: reference energy value if there is any (Default: None)
            stopping_threshold: stopping threshold for the training defined as mean(elocs)/std(elocs) (Default: 0.05)    
            store_model_freq: store the model only at epochs that is the multiplier of this value. Zero means nothing is stored. By default the model at the first and last epochs are saved. (Default: 0)
            observables: observables value to compute (Default: [])
            observable_freq: compute the observables only at epochs that is the multiplier of this value. Zero means nothing is stored. By default observables are calculated at the last epoch. (Default: 0)
            
        """
        self.hamiltonian = hamiltonian
        self.model = model
        self.observables = observables
        self.num_samples = num_samples
        self.reference_energy = reference_energy

        self.model_params = []
        self.observables_value = []

        
    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.model.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, train_data, val_data= None, epochs=1000, callbacks = [], batch_size=0):
        self.num_epochs = epochs
        self.batch_size = batch_size

        ## save weight
        self.store_model(0)

        if val_data is None:
            self.history = self.model.model.fit(train_data[0], train_data[1], batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        else:
            self.history = self.model.model.fit(train_data[0], train_data[1], validation_data= val_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        
        self.history = self.history.history
        epoch = len(self.history)

        self.store_model(epoch, last=True)
        
        for callback in callbacks:
            if 'EnergyCallback' in callback.__class__.__name__:
                self.samples = callback.samples

        ## save the last data
        self.store_model(epoch, last=True)

        self.calculate_observables(epoch)            

    def get_overlap(self, true_wavefunction):
        pred_wavefunction = self.get_wave_function().numpy()
        return np.abs(np.dot(pred_wavefunction.T, true_wavefunction)) ** 2 / np.sum(np.abs(pred_wavefunction) ** 2)

    def get_local_energy(self, samples):
        """
            Calculate local energy from a given samples
            $E_{loc}(x) = \sum_{x'} H_{x,x'} \Psi(x') / \Psi(x)$
            In this part, we instead do $log(\Psi(x')) - log(\Psi(x))$
            Args:
                samples: samples that we want to calculate the local energy
            Return:
                The local energy of each given samples
        """
        ## Calculate $H_{x,x'}$
        hamiltonian = self.hamiltonian.calculate_hamiltonian_matrix(samples, samples.shape[0])
        ## Calculate $log(\Psi(x')) - log(\Psi(x))$
        lvd = self.hamiltonian.calculate_ratio(samples, self.model, samples.shape[0])

        ## Sum over x'
        eloc_array = tf.reduce_sum((tf.exp( lvd) * hamiltonian), axis=1, keepdims=True)

        return eloc_array

    def calculate_observables(self, epoch): 
        """
            Calculate observables if any.
            Args:
                epoch: epoch for log purposes
        """
        ### Get the probability $|\Psi(x)|^2$ from samples 
        confs, count_ = np.unique(self.samples, axis=0, return_counts=True)
        prob_out = count_ / len(self.samples)

        ### Calculate each observables
        value_map = {}
        for obs in self.observables: 
            obs_value = (obs.get_value_ferro(prob_out, confs), obs.get_value_antiferro(prob_out, confs))
            value_map[obs.get_name()] = obs_value 
            
        self.observables_value.append((epoch, value_map))
        
    def store_model(self, epoch, last=False):
        """
        Store the model parameters in model_params at each epoch based on store_model_freq if needed.
        First and last epoch always stored.
        Args:
            epoch: the epoch 
            last: to mark if it is the last epoch or not
        """
        if last or epoch == 0:
            self.model_params.append((epoch, self.model.get_parameters()))
        else:
            if self.store_model_freq != 0 and epoch % self.store_model_freq == 0:
                self.model_params.append((epoch, self.model.get_parameters()))


    def make_pickle_object(self):
        """
        Create pickle object to save.
        """
        temp_learner = copy.copy(self)
        ## pickle cannot save a tensorflow object so it needs to be removed
        temp_learner.model.make_pickle_object()
        return temp_learner

    def to_xml(self):
        str = ""
        str += "<learner_supervised>\n"
        str += "\t<params>\n"
        str += "\t\t<optimizer>%s</optimizer>\n" % self.optimizer
        str += "\t\t<lr>%.5f</lr>\n" % self.optimizer.get_config()['learning_rate']
        str += "\t\t<epochs>%d</epochs>\n" % self.num_epochs
        str += "\t\t<minibatch>%d</minibatch>\n" % self.batch_size
        str += "\t</params>\n"
        str += "</learner_supervised>\n"
        return str
