import numpy as np
from observable import Observable

class CorrelationZ(Observable):
    """
    this class used to define the correlationz observable defined as: $c^z_{l,m} = \sum_x p(x) x_l x_m$ 
    for ferromagnetic, it is defined as $c^f_{i,d_0, d_1} = 1/(d_1 - d_0) \sum_{l=d_0}^{d_1} c^z_{i,l} $
    for antiferromagnetic, it is defined as $c^a_{i,d_0, d_1} = 1/(d_1 - d_0 ) \sum_{l=d_0}^{d_1} (-1)^l c^z_{i,l} $

    for 1d, i is usually the particle at the first position.
    for > 1d, i is usually the particle in the middle to the rows or diagonals
    
    """

    def __init__(self, num_particles, position=0, d_0 = 0, d_1 = None):
        """
        Construct a CorrelationZ observable model.
        Args:
            num_particles: number of particles                
            position: the position i (default: 0) could be an array
            d_0 = value of d_0
            d_1 = vale of d_1
        """
        Observable.__init__(self, num_particles)
        self.position = position
        self.C_total = None
        self.d_0 = d_0
        self.d_1 = d_1

        if self.d_1 is None:
            self.d_1 = self.num_particles

    def get_value(self, prob, confs):
        """
        Calculates the array C^z_{l,m} where l comes from the position array from a given probability and configurations.
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
        Returns:
            The array C^z_{l,m}
        """
        if isinstance(self.position, int):
            self.position = [self.position]

        self.C_total = np.zeros((self.num_particles, self.num_particles))
        for part_1 in self.position:
            for part_2 in range(0, self.num_particles):
                if part_1 == part_2: continue
                for i, conf in enumerate(confs):
                    self.C_total[part_1][part_2] += prob[i] * conf[part_1] * conf[part_2]  

    def get_value_antiferro(self, prob, confs, position=None):
        """
        Calculates the C^A_{i,d_0,d_1} where i is specified by the position
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
            position: i, the position to calculate the particle's position (default: None, which will be set to one of the position in the array)
        Returns:
            the value C^A_{position, d_0, d_1}
        """
        self.get_value(prob, confs)

        if position is None:
            position = self.position[0]

        data_temp = self.C_total[position]

        cz_antiferro = 0.

        for i in range(self.d_0, self.d_1):
          cz_antiferro += ((-1) ** i) * data_temp[i]

        ## if position is inside the range ignore it
        if position >= self.d_0 and position <= self.d_1:
            cz_antiferro /= self.d_1 - self.d_0 - 1
        else:
            cz_antiferro /= self.d_1 - self.d_0

        return cz_antiferro

    def get_value_ferro(self, prob, confs, position=None):
        """
        Calculates the C^F_{i,d} where i is specified by the position
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations
            position: i, the position to calculate the particle's position (default: None, which will be set to one of the position in the array)
        Returns:
            the value C^F_{position, d}
        """
        self.get_value(prob, confs)

        if position is None:
            position = self.position[0]

        data_temp = self.C_total[position][self.d_0:self.d_1]

        cz_ferro = np.sum(data_temp)

        ## if position is inside the range ignore it
        if position >= self.d_0 and position <= self.d_1:
            cz_ferro /= self.d_1 - self.d_0 - 1
        else:
            cz_ferro /= self.d_1 - self.d_0

        return cz_ferro
 
    def get_name(self):
        """
        Return a string name of the observable
        """
        return 'Cz'
