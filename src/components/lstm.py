import torch as tc
from src.components.network_component import NetworkComponent


class LSTM(NetworkComponent):
    """
    Class for the implementation of the LSTM modules
    """
    def __init__(self, input_dim, output_dim, n_hidden_layers: int = 32, h_0=None, c_0=None, *args, **kwargs):
        self.n_hidden_layers = n_hidden_layers
        super(LSTM, self).__init__(input_dim, output_dim)
        self.h_0 = self.h_n = h_0
        self.c_0 = self.c_n = c_0
        # TODO: construct model here;
        self.network = tc.nn.LSTM(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=n_hidden_layers,
            **kwargs                    # any additional keyword arguments will be passed to the LSTM
        )
        pass

    def input(self, params, **kwargs):
        """
        Method for parsing input.
        :param params: input tensor
        :return: output tensor; or [output tensor, (h_n, c_n)] if not simplified results
        """
        simplified_result = "simplified_result" not in kwargs or bool(kwargs["simplified_result"])
        # TODO: implement
        result, (h_n, c_n) = self.network(params, (self.h_0, self.c_0))
        self.h_n = h_n
        self.c_n = c_n
        return result if simplified_result else result, (h_n, c_n)

    def parameters(self):
        """
        Method for obtaining parameters of model for learning.
        :return: model parameters
        """
        # TODO: implement with model
        return self.network.all_weights

    def get_network_weights(self):
        """
        Method for obtaining parameter tensors.
        :return: h_n and c_n
        """
        return self.h_n, self.c_n
