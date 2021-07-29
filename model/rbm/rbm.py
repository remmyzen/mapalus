from model.model import Model

class RBM(Model):

    def __init__(self, num_visible, density=2):
        Model.__init__(self)
        self.num_visible = num_visible
        self.density = density
        self.num_hidden = int(self.num_visible * self.density)
        self.W = None
        self.bv = None
        self.bh = None
        self.connection = None

    def is_complex(self):
        return False

    def is_real(self):
        return False
