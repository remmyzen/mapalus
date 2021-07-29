import numpy as np
from observable import Observable

class CorrelationNNZ(Observable):
    """
    This class used to define the correlation z for next nearest neighbour
    For antiferromagnetic: 
        $C^A_{NNN}$ = \sum{x} p(x) (-1 / N-2 \sum_i^{N-2} x_i x_{i+2})
    For ferromagnetic: 
        $C^F_{NNN}$ = \sum{x} p(x) (-1 / N-2 \sum_i^{N-2} x_i x_{i+2})

    
    """

    def __init__(self, chain):
        """
        Construct a CorrelationZ observable model.
        Args:
            num_particles: number of particles
            
        """
        if not chain.next_nearest:
            print('WARNING! Not a graph with next nearest neighbour connection')
    
        self.chain = chain

        Observable.__init__(self, self.chain.num_points) 


    def get_value_antiferro(self, prob, confs):
        """
        Calculates the $C^A_NNN$
        $C^A_NNN$ = \sum{x} p(x) (-1 / N-2 \sum_i^{N-2} x_i x_{i+2})
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
        Returns:
            the value $C^A_NNN$
        """

        cz_antiferro = 0.
        
        for ii, conf in enumerate(confs):
            sum = 0
            for (aa, bb) in self.chain.bonds_next:
                sum += conf[aa] * conf[bb]
            sum *= -1
            sum /= len(self.chain.bonds_next)
            cz_antiferro += prob[ii] * sum

        return cz_antiferro

    def get_value_ferro(self, prob, confs, position=None):
        """
        Calculates the $C^F_NNN$
        $C^F_NNN$ = \sum{x} p(x) (-1 / N-2 \sum_i^{N-2} x_i x_{i+2})
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
        Returns:
            the value $C^F_NNN$
        """

        cz_ferro = 0.
        
        for ii, conf in enumerate(confs):
            sum = 0
            for (aa, bb) in self.chain.bonds_next:
                sum += conf[aa] * conf[bb]
            sum /= len(self.chain.bonds_next)
            cz_ferro += prob[ii] * sum

        return cz_ferro
 
    def get_name(self):
        """
        Return a string name of the observable
        """
        return 'CNNz'
