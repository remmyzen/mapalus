from model.mlp import MLP
import tensorflow as tf
from tensorflow import keras
import copy
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MLPRealPosSupervised(MLP):
    """
    This class is used to define a multilayer perceptron with real and 
    positive wavefunction for supervised classification. Therefore, there is only one output without any activation function
    wavefunction. 
    """

    def __init__(self, num_visible, num_hidden=[256], activation_hidden='tanh', activation_output=None, num_expe=None, use_bias=True, freeze_layer=[]):
        """
        Construct an multilayer perceptron model for real positive wavefunction.
        
        Args:
            num_visible: number of input nodes in the input layer.
            num_hidden: number of hidden nodes in the hidden layer.
            activation_hidden: the activation in the hidden layer.
            activation_output: the activation in the output layer.
            num_expe: number of experiment to determine the seed.
            use_bias: whether to use bias or not.
            freeze_layer: a list to freeze the weights or the biases.
                          where the index 0 and 1 refers to the weights and biases
                          from input layer to the first hidden layer, respectively,
                          and so on. (TODO: not yet implemented)
        """
    
        MLP.__init__(self, num_visible, num_hidden)
        self.num_expe = num_expe
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.use_bias = use_bias
        self.freeze_layer = freeze_layer
        self.num_expe = num_expe

        ## Set the same seed
        if num_expe is not None:
            np.random.seed(num_expe)
            tf.random.set_seed(num_expe)

        self.build_model()

    def build_model(self):
        """
        Create the model with Keras
        """
        inputs = tf.keras.layers.Input(shape=(self.num_visible,))  
        for ii in range(self.num_layer):
            if ii == 0:
                hidden = tf.keras.layers.Dense(self.num_hidden[ii], activation=self.activation_hidden, use_bias=self.use_bias)(inputs)
#                            , kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(inputs)
            else:
                hidden = tf.keras.layers.Dense(self.num_hidden[ii], activation=self.activation_hidden, use_bias=self.use_bias)(hidden)
#                           , kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001))(hidden)
        
        outputs = tf.keras.layers.Dense(1, activation=self.activation_output, use_bias = self.use_bias)(hidden)
        self.model = tf.keras.models.Model(inputs = inputs, outputs = outputs) 


    def sample(self, num_samples):
        """
            Sample from the multilayer perceptron from all probability.
            For now only supported from all probability
            TODO: Use metropolis for sampling

            Args:
                num_samples: the number of samples
        """ 
        xs = self.get_all_configurations()
        probs = self.get_all_probability()
        num_conf = 2 ** self.num_visible
        samples_index = np.random.choice(range(num_conf), num_samples, p=probs.numpy().flatten(), replace=True)        
        return(xs[samples_index, :])
        

    def get_all_probability(self):
        """
            Compute the probability |\psi(x) ** 2| / Z
        """ 
        return self.get_wave_function() ** 2 / tf.reduce_sum(self.get_wave_function() ** 2)
    
    def get_wave_function(self):
        """
            Get the full wave function \psi(x).
        """
        xs = self.get_all_configurations()
        wv = tf.exp(self.log_val(xs))       
        return wv

    def get_all_configurations(self):
        """
            Compute all of the possible configurations.    
        """
        confs = []
        for i in range(2 ** self.num_visible):
            conf_bin = format(i, '#0%db' % (self.num_visible + 2))
            ## configuration in binary -1 1
            conf = np.array([1 if c == '1' else -1 for c in conf_bin[2:]])
            confs.append(conf)
        return np.array(confs)

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


    def get_parameters(self):
        """
        Get the parameters for this model
        """
        if self.model is None:
            return self.params
        else:
            return self.model.get_weights()

    def set_parameters(self, params):
        """
        Set the parameters for this model for transfer learning or loading model purposes
        Args:
            params: the parameters to be set.
        """
        self.model.set_weights(params)

    def param_difference (self, first_param, last_param):
        """
        Calculate the difference between two parameters.
        This is equals to the sum of the mean squared difference of all parameters (weights and biases)
        """ 
        sum_diff = 0.
        for (par1, par2) in zip(first_param[1], last_param[1]):
            sum_diff += np.mean((par1 - par2) ** 2)

        return sum_diff
    
    def visualize_param (self, params, path):
        """
        Visualize every parameters
        Args:
            params: the parameters that visualize
            path: the path to save the visualization
        """
        epoch = params[0]
        for ii, param in enumerate(params[1]):
            ## Reshape for bias
            if len(param.shape) == 1:
                param = np.reshape(param, (param.shape[0],1))

            plt.figure()
            if ii % 2 == 0:
                plt.title("Weight layer %d at epoch %d" % (ii + 1, epoch))
            else:
                plt.title("Bias layer %d at epoch %d" % (ii + 1, epoch))
            plt.imshow(param, cmap='hot', interpolation='nearest')
            plt.xticks(np.arange(0, param.shape[1], 1.0))
            plt.yticks(np.arange(0, param.shape[0], 1.0))
            plt.colorbar()
            plt.tight_layout()
            if ii % 2 == 0:
                plt.savefig(path + '/weight-layer-%d-%d.png' % (ii+1, epoch))
            else:
                plt.savefig(path + '/bias-layer-%d-%d.png' % (ii+1, epoch))
            plt.close()

    def get_name(self):
        """
        Get the name of the model
        """
        hidden_layer_str = '-'.join([str(hid) for hid in self.num_hidden])
        return 'mlprealpossupervised-%s' % (hidden_layer_str)
           
    def make_pickle_object(self):
        """
        Tensorflow object cannot be pickled so needs to be handled
        save the last param first and make it none
        """
        self.params = self.get_parameters()
        self.model = None 

    def __str__(self):
        return 'MLPRealPositiveSupervised %s' % (self.num_hidden)

    def to_xml(self):
        stri = ""
        stri += "<model>\n"
        stri += "\t<type>MLPRealPositiveSupervised</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<num_visible>%d</num_visible>\n" % self.num_visible
        stri += "\t\t<num_hidden>%s</num_hidden>\n" % self.num_hidden
        stri += "\t\t<activation_output>%s</activation_output>\n" % str(self.activation_output)
        stri += "\t\t<activation_hidden>%s</activation_hidden>\n" % str(self.activation_hidden)
        stri += "\t\t<use_bias>%s</use_bias>\n" % str(self.use_bias)
        stri += "\t\t<num_expe>%s</num_expe>\n" % str(self.num_expe)
        stri += "\t\t<freeze_layer>%s</freeze_layer>\n" % str(self.freeze_layer)
        stri += "\t</params>\n"
        stri += "</model>\n"
        return stri

