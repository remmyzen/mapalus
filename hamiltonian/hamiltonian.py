import numpy as np

class Hamiltonian(object):
    """
    Base class for Hamiltonian.

    This class defines the hamiltonian of the quantum many-body system.
    You must define how to get the Hamiltonian matrix.
    """

    def __init__(self, graph):
        self.graph = graph
        self.hamiltonian = None

        ## Pauli matrices
        self.SIGMA_X = np.array([[0, 1], [1, 0]])
        self.SIGMA_Y = np.array([[0, -1j], [1j, 0]])
        self.SIGMA_Z = np.array([[1, 0], [0, -1]])


    # Calculates the Hamiltonian matrix from list of samples. Returns a tensor.
    def calculate_hamiltonian_matrix(self, samples, num_samples):
        # implemented in subclass
        raise NotImplementedError

    def calculate_ratio(self, samples, machine, num_samples):
        # implemented in subclass
        raise NotImplementedError

    def diagonalize(self):
        # implemented in subclass
        raise NotImplementedError

    def diagonalize_sparse(self):
        # implemented in subclass
        raise NotImplementedError

    def get_gs_energy(self):
        """
        Get ground state energy $E_0$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize()!")
        else:
            return np.real(np.min(self.eigen_values))

    def get_gs(self):
        """
        Get ground state $\Psi_{GS}$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize()!")
        else:
            return self.eigen_vectors[:, np.argmin(self.eigen_values)]

    def get_gs_probability(self):
        """
        Get ground state probability $|\Psi_{GS}|^2$
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize()!")
        else:
            return np.abs(self.get_gs()) ** 2

    def get_full_hamiltonian(self):
        """
        Get the full Hamiltonian matrix H
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with diagonalize()!")
        else:
            return self.hamiltonian

    def get_gs_local_energy(self):
        """
        Get the ground state local energy
        """
        if self.hamiltonian is None:
            print("Solve hamiltonian first with _diagonalize_hamiltonian!")
        else:
            gs = self.get_gs()
            eloc = np.matmul(self.hamiltonian, gs) / gs
            return eloc
    

