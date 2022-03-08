class NetworkComponent(object):
    """
    Base class for all Network components
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, params):
        raise NotImplementedError("This method must be overwritten!")

    def backward(self, params):
        raise NotImplementedError("This method must be overwritten!")

    def input(self, params):
        raise NotImplementedError("This method must be overwritten!")
