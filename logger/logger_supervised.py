from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class LoggerSupervised(object):
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
        self.visualize_overlap(learner)
        self.visualize_loss(learner)
        self.visualize_params(learner)

        ## Calculate observables
        if learner.observables is not None and len(learner.observables) > 0:
            self.calculate_observables(learner) 

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
                f.write('num,ground_energy,ground_energy_std,epoch,time,overlap,loss,observables\n')
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

        new_samples = learner.samples
        learner.samples_observ = new_samples

        ## Calculate new energy with new samples
        new_energy = learner.get_local_energy(new_samples)
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
        if 'eloc_mean' not in learner.history.keys():
            print('======= No eloc in history!')
        else:
            plt.figure()
            plt.title('Energy vs Iteration, %s, %s' % (str(learner.hamiltonian), str(learner.model)))
            plt.ylabel('Energy')
            plt.xlabel('Iteration #')
            ground_energy = np.array(learner.history['eloc_mean'])
            ground_energy_std = np.array(learner.history['eloc_std'])
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

    def visualize_overlap(self, learner):
        """
        Visualize the overlap evolution
        Args:
            learner: the learner object
        """
        if 'overlap' not in learner.history.keys():
            print('======= No overlap in history!')
        else:
            plt.figure()
            plt.title('Overlap vs Iteration, %s, %s' % (str(learner.hamiltonian), str(learner.model)))
            plt.ylabel('Overlap')
            plt.xlabel('Iteration #')
            overlap = np.array(learner.history['overlap'])
            plt.plot(range(len(overlap)), overlap, label='overlap')

            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(self.result_path + '/iter-overlap.png')
            plt.close()

    def visualize_loss(self, learner):
        """
        Visualize the loss evolution
        Args:
            learner: the learner object
        """
        plt.figure()
        plt.title('Loss vs Iteration, %s, %s' % (str(learner.hamiltonian), str(learner.model)))
        plt.ylabel('Loss')
        plt.xlabel('Iteration #')
        loss = np.array(learner.history['loss'])
        plt.plot(range(len(loss)), loss, label='train_loss')
        if 'val_loss' in learner.history.keys():
            val_loss = np.array(learner.history['loss'])
            plt.plot(range(len(val_loss)), val_loss, label='val_loss')

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.result_path + '/iter-loss.png')
        plt.close()

    def write_logs(self, learner):
        """
        Write logs for each epoch and the summary of experiment logs
        Args:
            learner: the learner object
        """
        losses = learner.history['loss']
    
        ground_energys = np.zeros_like(losses)
        ground_energy_stds = np.zeros_like(losses)
        if 'eloc_mean' in learner.history.keys():
            ground_energys = learner.history['eloc_mean']
            ground_energy_stds = learner.history['eloc_std']
            if learner.reference_energy is None:
                rel_errors = [0] * len(ground_energys)
            else:
                rel_errors = np.abs((ground_energys - learner.reference_energy) / learner.reference_energy) 
        
        overlaps = np.zeros_like(losses)
        if 'overlap' in learner.history.keys():
            overlaps = learner.history['overlap']


        times = np.zeros_like(losses)
        if 'time' in learner.history.keys():
            times = learner.history['time']

        filename = 'logs.csv'
        path = self.result_path + filename
        ff = open(path, 'w')
        ff.write('epoch,ground_energy,ground_energy_std,overlaps,losses,rel_error,time\n')
        for ep, (ge, ges, ov, lo, re, ti) in enumerate(
                zip(ground_energys, ground_energy_stds, overlaps, losses, rel_errors, times)):
            ff.write('%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (ep, ge, ges, ov, lo, re, ti))
        ff.close()

        ## Write to experiment_logs
        filename = 'experiment_logs.csv'
        path = self.parent_result_path + filename
        ff = open(path, 'a')
        ff.write('%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%s\n' % (
        self.num_experiment, ground_energys[-1], ground_energy_stds[-1], len(ground_energys) - 1, np.sum(times), overlaps[-1], losses[-1], self.observ_str))
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

