import numpy as np
from observable import Observable

class CorrelationNZ(Observable):
    """
    This class used to define the correlation z for nearest neighbour
    For antiferromagnetic: 
        $C^A_{NN}$ = \sum{x} p(x) (-1 / N \sum_i^{N} x_i x_{i+1})
    For ferromagnetic: 
        $C^F_{NN}$ = \sum{x} p(x) (1 / N \sum_i^{N} x_i x_{i+1})

    
    """

    def __init__(self, chain):
        """
        Construct a CorrelationZ observable model.
        Args:
            num_particles: number of particles
            
        """
    
        self.chain = chain

        Observable.__init__(self, self.chain.num_points) 


    def get_value_antiferro(self, prob, confs):
        """
        Calculates the $C^A_NN$
        $C^A_{NN}$ = \sum{x} p(x) (-1 / N \sum_i^{N} x_i x_{i+1})
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
        Returns:
            the value $C^A_NN$
        """

        cz_antiferro = 0.
        
        for ii, conf in enumerate(confs):
            sum = 0
            for (aa, bb) in self.chain.bonds:
                sum += conf[aa] * conf[bb]
            sum *= -1
            sum /= len(self.chain.bonds)
            cz_antiferro += prob[ii] * sum

        return cz_antiferro

    def get_value_ferro(self, prob, confs, position=None):
        """
        Calculates the $C^F_NNN$
        $C^F_{NN}$ = \sum{x} p(x) (1 / N \sum_i^{N} x_i x_{i+1})
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
        Returns:
            the value $C^F_NN$
        """

        cz_ferro = 0.
        
        for ii, conf in enumerate(confs):
            sum = 0
            for (aa, bb) in self.chain.bonds:
                sum += conf[aa] * conf[bb]
            sum /= len(self.chain.bonds)
            cz_ferro += prob[ii] * sum

        return cz_ferro
 
    def get_name(self):
        """
        Return a string name of the observable
        """
        return 'CNz'
