import tensorflow as tf
from hamiltonian import Hamiltonian
import itertools
import numpy as np
import scipy
import scipy.sparse.linalg

class Heisenberg(Hamiltonian):
    """
    This class is used to define Heisenberg XYZ model.
    Nearest neighbor interaction along x-, y- and z-axis with magnitude J_x, J_y and J_z, respectively.
    
    $H_H = \sum_{<i,j>} J_z \sigma^z_i \sigma^z_j + J_y \sigma^y_i \sigma^y_j + J_x \sigma^x_i \sigma^x_j$
    """

    def __init__(self, graph, jx=1.0, jy=1.0, jz=1.0, total_sz = None):
        """
        Construct an Heisenberg XYZ model.
            
        Args:
            jx: magnitude of the nearest neighbor interaction along x-axis
            jy: magnitude of external magnetic field along y-axis
            jz: magnitude of external magnetic field along z-axis
            total_sz: total_sz if we want to restrict the hilbert space
        """ 

        Hamiltonian.__init__(self, graph)
        self.jx = jx
        self.jy = jy
        self.jz = jz
        self.total_sz = total_sz

    def calculate_hamiltonian_matrix(self, samples, num_samples):
        """
        Calculate the Hamiltonian matrix $H_{x,x'}$ from a given samples x.
        Only non-zero elements are returned.

        Args:
            samples: The samples 
            num_samples: number of samples

        Return:
            The Hamiltonian where the first column contains the diagonal, which is  $-J_z \sum_{i,j} x_i x_j$.
            The rest of the column contains the off-diagonal, which is (J_x - J_y * x_i * x_j). 
            Therefore, the number of column equals the number of particles + 1 and the number of rows = num_samples
        """
        
        diagonal = tf.zeros((num_samples,))
        off_diagonal = None
        for (s, s_2) in self.graph.bonds:
            # Diagonal element of the hamiltonian
            # $J_z \sum_{i,j} x_i x_j$
            diagonal += self.jz * samples[:,s] * samples[:,s_2]

            # Off diagonal element of the hamiltonian
            # $(J_x - J_y * x_i * x_j)$
            if off_diagonal is None:
                off_diagonal = (self.jx - self.jy * samples[:,s] * samples[:, s_2])
                off_diagonal = tf.reshape(off_diagonal, (num_samples, 1))
            else:
                temp = (self.jx - self.jy * samples[:,s] * samples[:, s_2])
                temp = tf.reshape(temp, (num_samples, 1))
                off_diagonal = tf.concat((off_diagonal, temp ),
                                                 axis=1)

        diagonal = tf.reshape(diagonal, (num_samples, 1))

        hamiltonian = tf.concat((diagonal, off_diagonal), axis=1)

        return hamiltonian


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
            In the Heisenberg model, this corresponds x' where two adjacent spins are flipped. 
            Therefore, the number of column equals the number of particles + 1 and the number of rows = num_samples
        
        """

        lvd = model.log_val_diff(samples, samples)
        for (s, s2) in self.graph.bonds:
            ## Flip 2 adjacent spin
            flipped_s =  tf.reshape(samples[:,s] * -1 , (num_samples, 1))    
            flipped_s2 = tf.reshape(samples[:,s2] * -1, (num_samples, 1))
            if s == 0:
                new_config = tf.concat((flipped_s, flipped_s2, samples[:,s2+1:]), axis = 1)
            elif s2 == self.graph.num_points-1:  
                new_config = tf.concat((samples[:, :s], flipped_s, flipped_s2), axis = 1)
            else:
                new_config = tf.concat((samples[:, :s], flipped_s, flipped_s2, samples[:,s2+1:]), axis = 1)

            lvd = tf.concat((lvd, model.log_val_diff(new_config, samples)), axis=1)
        return lvd

    def diagonalize(self):
        """
        Diagonalize hamiltonian with exact diagonalization.
        Only works for small systems (<= 10)!
        """
        num_particles = self.graph.num_points
        ## Initialize zeroes hamiltonian
        H = np.zeros((2 ** num_particles, 2 ** num_particles), dtype='complex')

        ## Calculate interaction energy x-axis
        for i, a in self.graph.bonds:
            togg_vect = np.zeros(num_particles)
            togg_vect[i] = 1
            togg_vect[a] = 1

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_X)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.jx * temp

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Y)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.jy * temp

            temp = 1
            for j in togg_vect:
                if j == 1:
                    temp = np.kron(temp, self.SIGMA_Z)
                else:
                    temp = np.kron(temp, np.identity(2))
            H += self.jz * temp

        ## Filter total sz
        if self.total_sz is not None:
            index = []
            num_confs = 2 ** num_particles
            for row in range(num_confs):
                ## configuration in binary 0 1
                conf_bin = format(row, '#0%db' % (num_particles + 2))
                ## configuration in binary -1 1
                conf = [1 if c == '1' else -1 for c in conf_bin[2:]]
                
                if np.sum(conf) == self.total_sz:
                    index.append(row)
            
            H = H[index] 
            H = H[:, index]

        ## Calculate the eigen value
        self.eigen_values, self.eigen_vectors = np.linalg.eig(H)
        self.hamiltonian = H

    def diagonalize_sparse(self):
        """
        Diagonalize hamiltonian with exact diagonalization with sparse matrix.
        Only works for small (<= 20) systems!
        """
        num_particles = self.graph.num_points
        num_confs = 2 ** num_particles

        ## Constructing the COO sparse matrix
        row_ind = []
        col_ind = []
        data = []

        if self.total_sz is not None:
            index = []
            num_confs = 2 ** num_particles
            for row in range(num_confs):
                ## configuration in binary 0 1
                conf_bin = format(row, '#0%db' % (num_particles + 2))
                ## configuration in binary -1 1
                conf = [1 if c == '1' else -1 for c in conf_bin[2:]]
                
                if np.sum(conf) == self.total_sz:
                    index.append(row)
            index = np.array(index)            

        for row in range(num_confs):
            if self.total_sz is not None:
                if row not in index: 
                    continue
                row_map = np.where(index == row)[0][0]
            else:
                row_map = row

            ## configuration in binary 0 1
            conf_bin = format(row, '#0%db' % (num_particles + 2))
            ## configuration in binary -1 1
            conf = [1 if c == '1' else -1 for c in conf_bin[2:]]

            ## Diagonal = -J \sum SiSj
            row_ind.append(row_map)
            col_ind.append(row_map)
            total = 0
            for (i,j) in self.graph.bonds:
                total += conf[i] * conf[j]

            total *= self.jz

            data.append(total)

            for (i,j) in self.graph.bonds:
                ## flip i and j
                conf_temp = conf[:]
                conf_temp[i] *= -1
                conf_temp[j] *= -1

                col = int(''.join(['1' if a == 1 else '0' for a in conf_temp]), 2)
                if col == row: continue
                if self.total_sz is not None:
                    if col not in index: 
                        continue
                    col_map = np.where(index == col)[0][0]
                else:
                    col_map = col

                value = (self.jx - self.jy * conf[i] * conf[j])
                if value != 0:
                    row_ind.append(row_map)
                    col_ind.append(col_map)
                    data.append(value)

        row_ind = np.array(row_ind)
        col_ind = np.array(col_ind)
        data = np.array(data, dtype=float)

        mat_coo = scipy.sparse.coo_matrix((data, (row_ind, col_ind)))

        self.eigen_values, self.eigen_vectors = scipy.sparse.linalg.eigs(mat_coo, k=1, which='SR')
        self.hamiltonian = mat_coo
    
    def get_name(self):
        """ 
        Get the name of the Hamiltonian
        """
        if self.graph.pbc:
            bc = 'pbc'
        else:
            bc = 'obc'
        return 'heisenberg_%dd_%d_%.3f_%.3f_%.3f_%s' % (
        self.graph.dimension, self.graph.length, self.jx,
            self.jy, self.jz, bc)

    def __str__(self):
        return "Heisenberg %dD, jx=%.2f, jy=%.2f, jz=%.2f" % (self.graph.dimension, self.jx, self.jy, self.jz)

    def to_xml(self):
        str = ""
        str += "<hamiltonian>\n"
        str += "\t<type>heisenberg</type>\n"
        str += "\t<params>\n"
        str += "\t\t<jx>%d</jx>\n" % self.jx
        str += "\t\t<jy>%d</jy>\n" % self.jy
        str += "\t\t<jz>%d</jz>\n" % self.jz
        str += "\t</params>\n"
        str += "</hamiltonian>\n"
        return str
