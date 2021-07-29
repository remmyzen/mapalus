import numpy as np
from observable import Observable

class MagnetizationX(Observable):
    """
    this class used to define the magnetization observable along the x-axis defined as: 
    $<sigma_x> = \sum_{x,x'} \psi^*(x') <x'|\sigma_x|x>\psi(x) $ 
    $          = \sum_{x} p(x) \sum_{x'} \psi(x)/\psi(x') <x'|\sigma_x|x>$
    
    Returned value is <sigma_x> / (N - 1)
    
    """

    def __init__(self, num_particles):
        """
        Construct a CorrelationZ observable model.
        Args:
            num_particles: number of particles                
            position: the position i (default: 0) could be an array
            d_0 = value of d_0
            d_1 = vale of d_1
        """
        Observable.__init__(self, num_particles)
        self.value = None

    def get_value(self, probs, confs):
        """
        Calculates the array <sigma_x> a given probability and configurations.
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
        Returns:
            The array <sigma_x> / (N-1)
        """

        ### TODO: Need to change if using different assumption
        ### Compute wave function assuming that |p(x)|^2 = \psi(x)
        wavefunction = np.sqrt(probs)

        confs_mapping = {}
        ### Create a mapping from confs to wavefunction
        for ii, conf in enumerate(confs):
            result = int("".join("0" if i == -1 else "1" for i in conf),2)
            confs_mapping[result] = [probs[ii], wavefunction[ii]]


        ## Flip one by one
        total = 0
        for ii, conf in enumerate(confs):
            row = int("".join("0" if i == -1 else "1" for i in conf),2)
        
            total_inside = 0
            xor = 1
            for ii in range(self.num_particles):
                ## flipped the configuration
                #conf_flipped_bin = format(row ^ xor, '#0%db' % confs.shape[1])
                ## if flipped spin not in the mapping ignore
                if row ^ xor not in confs_mapping.keys(): continue
                    
                total_inside += confs_mapping[row ^ xor][1] / confs_mapping[row][1]
                # shift left to flip other bit locations
                xor = xor << 1
                
            total += total_inside * confs_mapping[row][0]
            
        self.value = np.real(total)

    def get_value_antiferro(self, prob, confs):
        """
            It returns value / N - 1
            It does not matter for antiferro and ferro.
        """
        if self.value is None:
            self.get_value(prob, confs)
        
        return self.value / self.num_particles

    def get_value_ferro(self, prob, confs, position=None):
        """
            It returns value / N - 1
            It does not matter for antiferro and ferro.
        """
        if self.value is None:
            self.get_value(prob, confs)
        return self.value / self.num_particles
 
    def get_name(self):
        """
            Return a string name of the observable
        """
        return 'Mx'
