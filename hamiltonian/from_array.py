import tensorflow as tf
from hamiltonian import Hamiltonian
import itertools
import numpy as np
import scipy
import scipy.sparse.linalg

class FromArray (Hamiltonian):
    """
    """

    def __init__(self, graph, array, seed = None):
        """
        """ 

        Hamiltonian.__init__(self, graph)

        self.seed = seed
        ## Set the same seed
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        

        ## Q * D * inv(Q)
        self.hamiltonian = array
        self.num_particles = int(np.log2(self.hamiltonian.shape[0]))
        
        ## [diagonal non-diagonal]
        self.shiftmat = 1 << np.arange(self.num_particles)[::-1]

        self.confs = []
        for i in range(2 ** self.num_particles):
            conf_bin = format(i, '#0%db' % (self.num_particles + 2))
            ## configuration in binary -1 1
            conf = np.array([1. if c == '1' else -1. for c in conf_bin[2:]], dtype='float32')
            self.confs.append(conf)
       
        self.confs = tf.convert_to_tensor(self.confs)

    def bin2int(self, b):
        return b.dot(self.shiftmat)

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        """
        Calculate the Hamiltonian matrix $H_{x,x'}$ from a given samples x.
        Only non-zero elements are returned.

        Args:
            samples: The samples 
            num_samples: number of samples

        Return:
            The Hamiltonian where the first column contains the diagonal, which is $-J \sum_{i,j} x_i x_j$.
            The rest of the column contains the off-diagonal, which is -h for every spin flip. 
            Therefore, the number of column equals the number of particles + 1 and the number of rows = num_samples
        """

        samples_numpy = samples.numpy()

        samples_numpy[samples_numpy == -1] = 0 
        index = np.array([self.bin2int(tes) for tes in samples_numpy[:]], dtype='int32')

        return tf.convert_to_tensor(self.hamiltonian[index,:], dtype=tf.float32)

    def calculate_ratio(self, samples, model, num_samples):
        """
       Calculate the ratio of \Psi(x') and \Psi(x) from a given x 
        as log(\Psi(x')) - log(\Psi(x))
       \Psi is defined in the model. 
        However, the Hamiltonian determines which x' gives non-zero.
        
        Args:
            samples: the samples x
            model: the model used to define \Psi
            num_samples: the number of samples
        Return:
            The ratio where the first column contains \Psi(x) / \Psi(x).
            The rest of the column contains the non-zero \Psi(x') / \Psi(x).
            In the Ising model, this corresponds x' where exactly one of spin x is flipped. 
            Therefore, the number of column equals the number of particles + 1 and the number of rows = num_samples
        
        """
        lvd = []
        for ii, sample in enumerate(samples):
            lvd.append(model.log_val_diff(self.confs, samples[ii:ii+1,:]))
        return tf.convert_to_tensor(lvd)[:,:,0]

    def diagonalize(self):
        """
        Diagonalize hamiltonian with exact diagonalization.
        Only works for small systems (<= 10)!
        """
        ## Calculate the eigen value
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.hamiltonian)


    def diagonalize_sparse(self):
        """
        Diagonalize hamiltonian with exact diagonalization with sparse matrix.
        Only works for small (<= 20) systems!
        """

        self.eigen_values, self.eigen_vectors = scipy.sparse.linalg.eigs(self.hamiltonian, k=1, which='SR')

    def get_name(self):
        """ 
        Get the name of the Hamiltonian
        """
        return 'fromarray_%d' % (
        self.num_particles)

    def __str__(self):
        return "FromArray %d" % (self.num_particles)

    def to_xml(self):
        str = ""
        str += "<hamiltonian>\n"
        str += "\t<type>from_array</type>\n"
        str += "\t<params>\n"
        str += "\t\t<seed>%s</seed>\n" % self.seed
        str += "\t\t<num_particles>%s</particles>\n" % self.num_particles
        str += "\t</params>\n"
        str += "</hamiltonian>\n"
        return str

