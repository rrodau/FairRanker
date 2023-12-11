import numpy as np

import torch
import torch.optim as optim
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
                 learning_rate=0.01,
                 max_steps=3000,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.944,
                 end_qids=20,
                 start_qids=10,
                 num_features=0,
                 num_fair_classes=0,
                 random_seed=None,
                 name="DirectRanker",
                 tensor_name_prefix="",
                 sess=None
                 ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.kernel_initializer = kernel_initializer
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.learning_rate_step_size = learning_rate_step_size
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.end_qids = end_qids
        self.start_qids = start_qids
        self.random_seed = random_seed
        self.name = name
        self.tensor_name_prefix = tensor_name_prefix
        self.sess = sess

        self.layers = nn.ModuleList()
        prev_nodes = num_features
        for num_neurons in self.hidden_layers:
            layer = nn.Linear(prev_nodes, num_neurons, bias=self.feature_bias)
            self.layers.append(layer)
            prev_nodes = num_neurons
        
        self.ranking_layer = nn.Linear(prev_nodes, 1, bias=False)

        self.ranking_layer_cls = nn.Linear(prev_nodes, 1, bias=False)



    def forward(self, x0, x1):
        mm0 = self.forward_extracted_features(x0)
        #print(mm0)
        mm1 = self.forward_extracted_features(x1)
        mm = self.calc_dist(mm0, mm1)
        mm = self.ranking_activation(self.ranking_layer(mm))
        #mm_cls = self.ranking_activation(self.ranking_layer_cls(mm0 / 2.))
        return mm
    
    """ def loss(self, mm, y0):
        ranking_loss = (y0 - mm) ** 2
        return torch.mean(ranking_loss)"""


    def forward_extracted_features(self, x):
        for layer in self.layers:
            x = self.feature_activation(layer(x))
        return x
    
    def calc_dist(self, mm0, mm1):
        return mm0 - mm1 #/ 2

        
