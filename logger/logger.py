from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Logger(object):
    """
    This class is used for logging purposes
    By default the data is stored in a folder name defined by result_path/hamiltonian_name/model_name/subpath/num_experiment
    """

    ## Global variables to set default result directory 
    default_result_path = './'
    ## Global variables to subpath of an experiment
    default_subpath = 'cold-start'

    def __init__(self, result_path=default_result_path, subpath=default_subpath):
        self.result_path = result_path
        self.subpath = subpath
        if self.subpath is None or self.subpath == '':
            self.subpath = Logger.default_subpath
        self.subpath = '/' + self.subpath + '/'

        ## String for log
        self.observ_str = ''

    def log(self, learner):
        """
        Main process to log
        Args:
            learner: the learner object that contains all of the information
        """
        print('===== Logging start')
        ## Get folder name and create the folder
        ## By default folder name is result_path/hamiltonian_name/model_name/subpath
        folder_name = self.get_folder_name(learner)
        self.make_base_path(folder_name)

        ## Print xml file
        self.print_model(learner)

        ## Visualization
        self.visualize_energy(learner)
        self.visualize_params(learner)

        ## Calculate observables
        if learner.observables is not None and len(learner.observables) > 0:
            self.calculate_observables(learner, learner.sampler.num_samples, learner.sampler.num_steps) 

        ## Calculate parameter difference
        self.calculate_param_difference(learner)

        ## Write logs
        self.write_logs(learner)

        ## Save model
        self.save_model(learner)
        print('===== Logging finish')

    def get_folder_name(self, learner):
        """
        Get folder name
        Args:
            learner: the learner object
        Return:
            the hamiltonian_name and model_name
        """
        hamiltonian_name = learner.hamiltonian.get_name()
        model_name = learner.model.get_name()
        return hamiltonian_name + '/' + model_name

    def make_base_path(self, name):
        """
        Create the folder path
        Args:
            name: the folder name
        """
        ## set the path and make the directory
        path = self.result_path + name + self.subpath
        self.make_directory(path)

        ## create an experiment log file
        self.make_experiment_logs(path)

        ## retrieve all subdirectory names to get the number of experiment
        dir_names = [int(f) for f in os.listdir(path) if os.path.isdir(path + f)]
        if len(dir_names) > 0:
            self.num_experiment = max(dir_names) + 1
        else:
            self.num_experiment = 0

        ## create the directory with the num_experiment
        next_dir = str(self.num_experiment) + '/'
        path = path + next_dir
        self.make_directory(path)
        self.result_path = path

    def make_experiment_logs(self, path):
        """
        Create an experiment logs for easy visualization.
        Args:
            path: the path to create the experiment_logs
        """
        if not os.path.exists(path + 'experiment_logs.csv'):
            with open(path + 'experiment_logs.csv', 'w') as f:
                f.write('num,ground_energy_window,ground_energy_std_window,variance,epoch,time,ground_energy,ground_energy_std,energy_obs,energy_std_obs,observables\n')
                f.close()
        self.parent_result_path = path

    def make_directory(self, path):
        """
        Create directory for a given path
        Args:
            path: the path to create
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, learner):
        """
        Save the model
        Args:
            learner: the learner object
        """
        filename = 'model.p'
        path = self.result_path + filename
        pickle.dump(learner.make_pickle_object(), open(path, 'wb'))
        print('===== Model saved in %s' % (path))

    def print_model(self, learner):
        """ 
        Print the model parameters as an XML file
        Args:
            learner: the learner object
        """
        filename = 'experiment.xml'
        path = self.result_path + filename
        ff = open(path, 'w')
        ff.write('<model>\n')
        ff.write(learner.to_xml())
        ff.write(learner.hamiltonian.graph.to_xml())
        ff.write(learner.hamiltonian.to_xml())
        ff.write(learner.model.to_xml())
        ff.write(learner.sampler.to_xml())
        ff.write('</model>\n')
        ff.close()

    def calculate_observables(self, learner, num_samples=1000, num_steps=100, num_division=1):
        """
        Calculate observables with more samples and steps
        Args:
            learner: the learner object
            num_samples: the number of samples
            num_steps: the number of steps
            num_division: divide the observable calculation for low memory computer
        """

        ## Set parameters for the sampler
        learner.sampler.set_num_samples(num_samples / num_division)

        if 'Metropolis' in learner.sampler.__class__.__name__:
            #num_steps = 5000
            num_steps = 1000
        elif learner.sampler.__class__.__name__ == 'Gibbs':
            num_steps = 1000        
    
        learner.sampler.set_num_steps(num_steps)

        ## Get new samples
        init_samples = learner.samples
        new_samples = learner.sampler.sample(learner.model, init_samples, int(num_samples / num_division))
        learner.samples_observ = new_samples

        ## Calculate new energy with new samples
        new_energy = np.real(learner.get_local_energy(new_samples))
        learner.energy_observ = new_energy

        filename = 'energy_observ.txt'
        ff = open(self.result_path + filename, 'w')
        ff.write('%.5f\n' % np.mean(new_energy))
        ff.write('%.5f\n' % np.std(new_energy))
        ff.close()
        self.observ_str += '%.5f,%.5f' % (np.mean(new_energy), np.std(new_energy))

        confs, count_ = np.unique(new_samples, axis=0, return_counts=True)
        prob_out = count_ / float(num_samples)
        ## Save the probability for small system
        if len(prob_out) < 5000:
            np.savetxt(self.result_path + 'probs.txt', prob_out)
            np.savetxt(self.result_path + 'confs.txt', confs)
            pickle.dump((confs, prob_out), open(self.result_path + 'probs.p', 'wb'))

        ## Save observables value
        for obs in learner.observables:
            obs_value = (obs.get_value_ferro(prob_out, confs), obs.get_value_antiferro(prob_out, confs))
            filename = obs.get_name() + '.txt'
            ff = open(self.result_path + filename, 'w')
            ff.write('%.5f,%.5f' % (obs_value[0], obs_value[1])) 
            ff.close()
            
            self.observ_str += ',%.5f,%.5f' % (obs_value[0], obs_value[1]) 

    def visualize_energy(self, learner):
        """
        Visualize the energy evolution
        Args:
            learner: the learner object
        """
        plt.figure()
        plt.title('Energy vs Iteration, %s, %s' % (str(learner.hamiltonian), str(learner.model)))
        plt.ylabel('Energy')
        plt.xlabel('Iteration num')
        ground_energy = np.array(learner.ground_energy)
        ground_energy_std = np.array(learner.ground_energy_std)
        plt.plot(range(len(ground_energy)), ground_energy, label='energy')
        plt.fill_between(range(len(ground_energy)), ground_energy - ground_energy_std,
                         ground_energy + ground_energy_std, alpha=0.4, color='red')
        if learner.reference_energy is not None:
            plt.axhline(y=learner.reference_energy, xmin=0, xmax=learner.num_epochs, linewidth=2, color='k',
                        label='Exact')

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.result_path + '/iter-energy.png')
        plt.close()

    def write_logs(self, learner):
        """
        Write logs for each epoch and the summary of experiment logs
        Args:
            learner: the learner object
        """
        ground_energys = learner.ground_energy
        ground_energy_stds = learner.ground_energy_std
        energy_windows = learner.energy_windows
        energy_windows_std = learner.energy_windows_std
        rel_errors = learner.rel_errors
        times = learner.times

        filename = 'logs.csv'
        path = self.result_path + filename
        ff = open(path, 'w')
        ff.write('epoch,ground_energy,ground_energy_std,ground_energy_window,rel_error,time\n')
        for ep, (ge, ges, gew, gest, re, ti) in enumerate(
                zip(ground_energys, ground_energy_stds, energy_windows, energy_windows_std, rel_errors, times)):
            ff.write('%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (ep, ge, ges, gew, gest, re, ti))
        ff.close()

        ## Write to experiment_logs
        filename = 'experiment_logs.csv'
        path = self.parent_result_path + filename
        ff = open(path, 'a')
        ff.write('%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s\n' % (
        self.num_experiment, energy_windows[-1], energy_windows_std[-1], ground_energy_stds[-1], len(ground_energys) - 1, np.sum(times), ground_energys[-1], ground_energy_stds[-1], self.observ_str))
        ff.close()

    def calculate_param_difference(self, learner):
        """ 
        Calculate the parameter difference
        Args:
            learner: the learner object
        """

        filename = 'weight_diff.txt'
        ff = open(self.result_path + filename, 'w')

        ## Get the first param
        first_param = learner.model_params[0]
        ## Get the last param
        last_param = learner.model_params[-1]
        ## Calculate the param difference
        diff = learner.model.param_difference(first_param, last_param)
        ff.write('%.8f' % (diff))
        ff.close()
         

    def visualize_params(self, learner):
        """
        Visualize the parameters of the first and the last one
        Args:
            learner: the learner object
        """
        ## Get the first param
        first_param = learner.model_params[0]
        ## Get the last param
        last_param = learner.model_params[-1]

        ## Visualize
        path = self.result_path + '/first_param/'
        self.make_directory(path)
        learner.model.visualize_param(first_param, path)
        path = self.result_path + '/last_param/'
        self.make_directory(path)
        learner.model.visualize_param(last_param, path)

