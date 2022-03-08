class LSTM(object):
    """
    Class for the implementation of the LSTM modules
    """
    def __init__(self, n_hidden_layers:int = 32):
        self.n_hidden_layers = n_hidden_layers
