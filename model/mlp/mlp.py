from model.model import Model

class MLP(Model):

    def __init__(self, num_visible, num_hidden=[256]):
        Model.__init__(self)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_layer = len(num_hidden)
        
    def is_complex(self):
        return False

    def is_real(self):
        return False
