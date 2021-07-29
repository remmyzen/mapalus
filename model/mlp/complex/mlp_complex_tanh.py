from model.mlp import MLP
import tensorflow as tf
import copy
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MLPComplexTanh(MLP):
    """
    This class is used to define a one-layer multilayer perceptron with tanh activation function
    for complex wavefunction.

    """
    

    def __init__(self, num_visible, density=2, initializer=None, use_bias=True, num_expe=None, freeze_pos=None):
        """
        Args:
            num_visible: number of visible
            density: the number hidden layer define as density * num_visible
            initializer: the initialization of the weights
            use_bias: use bias or not
            num_expe: number of experiment to determine the seed
        """
        MLP.__init__(self, num_visible, [num_visible * density])
        self.density = density
        self.initializer = initializer        
        self.use_bias = use_bias
        self.num_expe = num_expe
        
        self.freeze_pos = freeze_pos

        if num_expe is not None:
            np.random.seed(num_expe)
            tf.random.set_seed(num_expe)

        self.build_model()

    def build_model(self):
        """
        Build the RBM model
        """
        ## Randomly initialize the weights and bias
        self.random_initialize() 
        ## Create the model
        self.create_variable()

    def random_initialize(self):
        """
        Randomly initialize an array based on the initializer. Biases array are initialized zero.
        """
        self.num_hidden = int(self.num_hidden[0])
        ## Weights 1  (W1)
        #self.W1_array = self.initializer(size=(self.num_visible, self.num_hidden)) 
        self.W1_array = self.initializer(size=(self.num_visible, self.num_hidden)) + self.initializer(size=(self.num_visible, self.num_hidden)) * 1j
        ## Bias 1 (b1)
        self.b1_array = np.zeros((1, self.num_hidden))
        ## Weights 2  (W2)
        #self.W2_array = self.initializer(size=(self.num_hidden, 1)) 
        self.W2_array = self.initializer(size=(self.num_hidden, 1)) + self.initializer(size=(self.num_hidden, 1)) * 1j
        ## Bias 2 (b2)
        self.b2_array = np.zeros((1, 1))


    def create_variable(self):
        """
        Create model by creating a parameters variable, which is the weight, visible bias, hidden bias 
        """
        self.W1 = tf.Variable(tf.convert_to_tensor(value=self.W1_array.astype(np.complex64)), name="W1", trainable=True)
        self.b1 = tf.Variable(tf.convert_to_tensor(value=self.b1_array.astype(np.complex64)), name="b1", trainable=True)
        self.W2 = tf.Variable(tf.convert_to_tensor(value=self.W2_array.astype(np.complex64)), name="W2", trainable=True)
        self.b2 = tf.Variable(tf.convert_to_tensor(value=self.b2_array.astype(np.complex64)), name="b2", trainable=True)
        self.model = self
        self.trainable_weights = [self.W1, self.b1, self.W2, self.b2]

    def log_val(self, x):
        """
            Calculate log(\Psi(x)) 
            Args:
                x: the x
        """
        x = tf.cast(x, tf.complex64)
        z = tf.nn.tanh(tf.matmul(x, self.W1) + self.b1)
        
        theta = tf.matmul(z, self.W2) + self.b2

        return tf.math.log(theta)

    def log_val_diff(self, xprime, x):
        """
            Calculate log(\Psi(x')) - log(\Psi(x))
            Args:
                xprime: x'
                x: x
        """
        xprime = tf.cast(xprime, tf.complex64)
        x = tf.cast(x, tf.complex64)
        log_val_1 = self.log_val(xprime)
        log_val_2 = self.log_val(x)
        return log_val_1 - log_val_2

    def derlog(self, x):
        """
        Calculate $D_{W}(x) = D_{W} = (1 / \Psi(x)) * (d \Psi(x) / dW)$ where W can be the weights or the biases.
        """
        sample_size = x.shape[0]
        x = tf.cast(x, tf.complex64)

        ## Calculate theta = Wx + b
        z = tf.math.tanh(tf.matmul(x, self.W1) + self.b1)
        
        theta = tf.matmul(z, self.W2) + self.b2

        ## dy / dtheta
        dy_dtheta = tf.ones_like(theta)
        
        ## dw2
        D_W2 = tf.expand_dims(z,2) * tf.expand_dims(dy_dtheta, 1)

        ## db2
        D_b2 = dy_dtheta

        ## dy / dz
        dy_dz = tf.matmul(dy_dtheta, tf.transpose(self.W2)) * (1 - z ** 2)

        ## dw1 
        D_W1 = tf.expand_dims(x,2) * tf.expand_dims(dy_dz, 1)

        ## db1
        D_b1 = dy_dz


        if not self.use_bias:
            D_b1 = D_b1 * 0.0 
            D_b2 = D_b2 * 0.0

        D_W1 = tf.reshape(D_W1, (sample_size, *self.W1.shape)) / tf.expand_dims(theta,2)
        D_W2 = tf.reshape(D_W2, (sample_size, *self.W2.shape)) / tf.expand_dims(theta,2)
        D_b1 = tf.reshape(D_b1, (sample_size, *self.b1.shape)) / tf.expand_dims(theta,2)
        D_b2 = tf.reshape(D_b2, (sample_size, *self.b2.shape)) / tf.expand_dims(theta,2)

        if self.freeze_pos is not None:
            D_W1 = D_W1.numpy()
            D_W1[self.freeze_pos] = 0.
            D_W1 = tf.convert_to_tensor(D_W1)

        derlogs = [D_W1, D_b1, D_W2, D_b2]
        return derlogs

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
            plt.figure()
            if ii == 0:
                plt.title("Weight at epoch %d" % (epoch))
            elif ii == 1:
                plt.title("Visible Bias at epoch %d" % (epoch))
            elif ii == 2:
                plt.title("Hidden Bias at epoch %d" % (epoch))

            plt.imshow(np.real(param), cmap='hot', interpolation='nearest')
            plt.xticks(np.arange(0, param.shape[1], 1.0))
            plt.yticks(np.arange(0, param.shape[0], 1.0))
            plt.colorbar()
            plt.tight_layout()
            if ii == 0:
                plt.savefig(path + '/weight-layer-%d.png' % (epoch))
            elif ii == 1:
                plt.savefig(path + '/visbias-layer-%d.png' % (epoch))
            elif ii == 2:
                plt.savefig(path + '/hidbias-layer-%d.png' % (epoch))
            plt.close()

    def get_parameters(self):
        """
        Get the parameter of this model
        """
        return [self.W1.numpy(), self.b1.numpy(), self.W2.numpy(), self.b2.numpy()]

    def set_parameters(self, params):
        """
        Set the parameters for this model for transfer learning or loading model purposes
        Args:
            params: the parameters to be set.
        """
        self.W1.assign(params[0])
        self.b1.assign(params[1])
        self.W2.assign(params[2])
        self.b2.assign(params[3])

    def get_name(self):
        """
        Get the name of the model
        """
        return 'mlpcomplex-%d' % (self.num_hidden)

    def make_pickle_object(self):
        """
        Make pickle object for RBM
        Nothing to do for RBM
        """
        pass

    def is_complex(self):
        return True

    def __str__(self):
        return 'MLPComplex %d' % (self.num_hidden)

    def to_xml(self):
        stri = ""
        stri += "<model>\n"
        stri += "\t<type>mlp_complex</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<num_visible>%d</num_visible>\n" % self.num_visible
        stri += "\t\t<num_hidden>%d</num_hidden>\n" % self.num_hidden
        stri += "\t\t<density>%d</density>\n" % self.density
        stri += "\t\t<initializer>%s</initializer>\n" % str(self.initializer)
        stri += "\t\t<use_bias>%s</use_bias>\n" % str(self.use_bias)
        stri += "\t\t<num_expe>%s</num_expe>\n" % str(self.num_expe)
        stri += "\t</params>\n"
        stri += "</model>\n"
        return stri
