import torch
import torch.nn as nn


class DirectRanker(nn.Module):
    """
    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param feature_activation: tf function for the feature part of the net
    :param ranking_activation: tf function for the ranking part of the net
    :param feature_bias: boolean value if the feature part should contain a bias
    :param kernel_initializer: tf kernel_initializer
    :param start_batch_size: start value for increasing the sample size
    :param end_batch_size: end value for increasing the sample size
    :param learning_rate: learning rate for the optimizer
    :param max_steps: total training steps
    :param learning_rate_step_size: factor for increasing the learning rate
    :param learning_rate_decay_factor: factor for increasing the learning rate
    :param optimizer: tf optimizer object
    :param print_step: for which step the script should print out the cost for the current batch
    :param end_qids: end value for increasing the query size
    :param start_qids: start value for increasing the query size
    :param gamma: value how important the fair loss is
    """

    def __init__(self,
                 hidden_layers=[10, 5],
                 feature_activation=nn.Tanh(),
                 ranking_activation=nn.Tanh(),
                 feature_bias=True,
                 kernel_initializer=nn.init.normal_,
                 end_qids=20,
                 start_qids=10,
                 num_features=0,
                 random_seed=None,
                 name="DirectRanker",
                 ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.kernel_initializer = kernel_initializer
        self.end_qids = end_qids
        self.start_qids = start_qids
        self.random_seed = random_seed
        self.name = name

        self.layers = nn.ModuleList()
        prev_nodes = num_features
        for num_neurons in self.hidden_layers:
            layer = nn.Linear(prev_nodes, num_neurons, bias=self.feature_bias)
            self.kernel_initializer(layer.weight)
            self.layers.append(layer)
            prev_nodes = num_neurons
        
        self.ranking_layer = nn.Linear(prev_nodes, 1, bias=False)

        self.ranking_layer_cls = nn.Linear(prev_nodes, 1, bias=False)


    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        The forward function of the DirectRanker computes the rank score for
        the documents provided

        Parameters:
        - x0 (torch.Tensor): A document which needs to be ranked in relation to another document x1
        - x1 (torch.Tesnor): A document which needs to be ranked in relation to another document x0

        Returns:
        - torch.Tensor: Ranking score given to the paired documents. Values \in (0,-1] 
                        mean the document x1 is ranked higher whereas values \in (0, 1]
                        mean the document x0 is ranked higher.
        """
        # Extracted features from the first sample
        mm0 = self.forward_extracted_features(x0)
        # Extracted features from the second sample
        mm1 = self.forward_extracted_features(x1)
        # Subtract both extracted features
        mm = self.calc_dist(mm0, mm1)
        # Pass difference through the ranking layer
        mm = self.ranking_activation(self.ranking_layer(mm))
        return mm


    def forward_extracted_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the forward pass. It passes the input into the layers for the
        extracted feautres

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the layers in the
                            the network except the ranking layer
        
        Returns:
        - torch.Tensor: The extracted features of the document
        """
        for layer in self.layers:
            x = self.feature_activation(layer(x))
        return x
    

    def calc_dist(self, mm0: torch.Tensor, mm1: torch.Tensor) -> torch.Tensor:
        """
        A function which calculates the difference between two torch tensors
        these tensors are the extracted features

        Parameters:
        mm0 (torch.Tensor): The extracted features of the first document
        mm1 (torch.Tensor): The extracted features of the second document

        Returns:
        torch.Tensor: The difference between the two extracted features tensors
        """
        return mm0 - mm1 #/ 2
