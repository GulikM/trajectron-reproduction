from src.components.network_component import NetworkComponent


class ATTN(NetworkComponent):
    """
    Class for implementing the Attention Network.
    """
    def __init__(self, input_dim, output_dim, n_hidden_layers: int = 32):
        self.n_hidden_layers = n_hidden_layers
        super(ATTN, self).__init__(input_dim, output_dim)
        # TODO: construct model here
        pass

    def input(self, params, **kwargs):
        """
        Method for parsing input.
        :param params: input tensor
        :return: output tensor
        """
        # TODO: implement
        pass

    def parameters(self):
        """
        Method for obtaining parameters of model for learning.
        :return: model parameters
        """
        # TODO: implement with model
        pass