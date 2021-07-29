import numpy as np
from observable import Observable

class MagnetizationZSquare(Observable):
    """
    this class used to define the magnetization z square observable defined as: 
    for ferromagnetic, it is defined as $M^Z_F = 1/N \sum_{l=1}^{N} x_l$
    for antiferromagnetic, it is defined as $M^Z_A = 1/N \sum_{l=1}^{N} (-1)^(l+1) x_l$ 
    """


    def __init__(self, num_particles, dimension):
        """
        Construct a MagnetizationZSquare observable model.
        Args:
            num_particles: number of particles                
        """ 
        Observable.__init__(self, num_particles)
        self.dimension = dimension

    def get_value_antiferro(self, prob, confs):
        """
        Calculate value $M^Z_A = 1/N \sum_{l=1}^{N} (-1)^(l+1) x_l$ 
        Different treatment for different dimension
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations

        TODO: generalise it

        """
        M = 0
        if self.dimension == 1:
            for i, conf in enumerate(confs):
                temp = 0
                for j, c in enumerate(conf):
                    temp += ((-1) ** j) * c
                M += prob[i] * (temp ** 2)

        elif self.dimension == 2:
            for i, conf in enumerate(confs):
                temp  = 0
                mult = -1
                for j, c in enumerate(conf):
                    if j % int(np.sqrt(self.num_particles)) != 0:
                        mult *= -1
                    temp += mult * c
                M += prob[i] * (temp ** 2)
        elif self.dimension == 3:
            for i, conf in enumerate(confs):
                temp  = 0
                mult = -1
                it = 0
                for j in range(int(self.num_particles ** (1. / 3))):
                    if j > 0: mult *= -1
                    for k in range(int(self.num_particles ** (1. / 3))):
                        for l in range(int(self.num_particles ** (1. / 3))):
                            if l > 0: mult *= -1
                            temp += mult * conf[it]
                            it +=1
                M += prob[i] * (temp ** 2)

        return M / (self.num_particles ** 2)

    def get_value_ferro(self, prob, confs):
        """
        Calculate value $M^Z_F = 1/N^2 \sum_{l=1}^{N}  x_l$ 
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations

        """
        M = 0
        for i, conf in enumerate(confs):
            M += prob[i] * (np.sum(conf) ** 2)

        return M / (self.num_particles ** 2)

    def get_value_ferro_sz0(self, prob, confs):
        """
        Value for ferromagnetic with total sz = 0.
        It can be used to detect +1 +1 +1 ... -1 -1 -1 and -1 -1 -1 -1 ... +1 +1 +1
        Calculate value $M^Z_F = 1/N^2 \sum_{l=1}^{N}  x_l$ 
        Args:
            prob: p(x) probabality for the configurations
            confs: the list of x configurations

        """
        M = 0
        domain_wall = self.num_particles // 2
        for i, conf in enumerate(confs):
            M += prob[i] * ((np.sum(conf[:domain_wall]) - np.sum(conf[domain_wall:])) ** 2)

        return M / (self.num_particles ** 2)


    def get_name(self):
        return 'Mz2'
