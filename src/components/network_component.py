class NetworkComponent(object):
    """
    Base class for all Network components
    """
    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def input(self, params, **kwargs):
        raise NotImplementedError("This method must be overwritten!")

    def parameters(self):
        raise NotImplementedError("This method must be overwritten!")
