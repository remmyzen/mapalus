class Model(object):

    def __init__(self):
        self.num_visible = 0

    def log_val(self, v):
        pass

    def log_val_diff(self, v1, v2):
        pass

    def derlog(self, v, size):
        pass

    def get_parameters(self):
        pass

    def visualize_param(self):
        pass

    def is_complex(self):
        return False

    def is_probability(self):
        return False
