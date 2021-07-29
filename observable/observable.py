
class Observable(object):
    """
    Base class for Observable.

    This class defines an observable.
    TODO: Non diagonal Hamiltonian
    """
        
    def __init__(self, num_particles):
        self.num_particles = num_particles

    def get_value_ferro(self):
        return None

    def get_value_antiferro(self):
        return None

    def get_name(self):
        return None
