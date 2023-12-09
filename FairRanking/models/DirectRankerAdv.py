import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from FairRanking.models.BaseDirectRanker import BaseDirectRanker
from FairRanking.models.flip_gradient import flipGradient
from FairRanking.models.DirectRanker import DirectRanker

class DirectRankerAdv(BaseDirectRanker):
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
                 bias_layers=[50, 20, 2],
                 feature_activation=nn.Tanh,
                 ranking_activation=nn.Tanh,
                 feature_bias=True,
                 kernel_initializer=nn.init.normal_,
                 start_batch_size=100,
                 end_batch_size=500,
                 learning_rate=0.01,
                 max_steps=3000,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.994,
                 optimizer=optim.Adam,
                 print_step=0,
                 end_qids=300,
                 start_qids=10,
                 random_seed=42,
                 name="DirectRankerAdv",
                 gamma=1,
                 dataset=None,
                 noise_module=False,
                 noise_type="sigmoid_full",
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1,
                 whiteout_lambda=1,
                 num_features=0,
                 num_fair_classes=0,
                 save_dir=None):
        super().__init__(hidden_layers=hidden_layers, dataset=dataset,
                    feature_activation=feature_activation, ranking_activation=ranking_activation,
                    feature_bias=feature_bias, kernel_initializer=kernel_initializer,
                    start_batch_size=start_batch_size, end_batch_size=end_batch_size,
                    learning_rate=learning_rate, max_steps=max_steps,
                    learning_rate_step_size=learning_rate_step_size,
                    learning_rate_decay_factor=learning_rate_decay_factor, optimizer=optimizer,
                    print_step=print_step,
                    end_qids=end_qids, start_qids=start_qids, random_seed=random_seed, name=name, gamma=gamma,
                    noise_module=noise_module, noise_type=noise_type, whiteout=whiteout,
                    uniform_noise=uniform_noise,
                    whiteout_gamma=whiteout_gamma, whiteout_lambda=whiteout_lambda, num_features=num_features,
                    num_fair_classes=num_fair_classes, save_dir=save_dir)
        self.bias_layers = bias_layers

        # Feature Layers
        self.feature_layers = nn.ModuleList()
        prev_neurons = num_features
        for num_neurons in hidden_layers:
            layer = nn.Linear(prev_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.feature_layers.append(layer)
            prev_neurons = num_neurons

        # debias Layers
        self.debias_layers = nn.ModuleList()
        for num_neurons in bias_layers:
            layer = nn.Linear(prev_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.debias_layers.append(layer)
            prev_neurons = num_neurons

        # Ranking Layers
        # auxillary ranker?
        self.ranking_layer = nn.ModuleList()
        self.ranking_layer.append(nn.Linear(prev_neurons, num_neurons))
        #prev_neurons = num_neurons
        
    def forward(self, x0, x1):
        # Apply noise module if enabled
        if self.noise_module:
            in_0, in_1 = self.create_noise_module(x0, x1)
        else:
            in_0, in_1 = x0, x1

        # Process through feature layers
        in_0 = self.forward_extracted_features(in_0)
        in_1 = self.forward_extracted_features(in_1)

        extracted_features = in_0

        # Apply gradient reversal for bias handling
        nn_bias0 = flipGradient(in_0)
        nn_bias1 = flipGradient(in_1)

        nn_bias0 = self.forward_debias_layers(nn_bias0)
        nn_bias1 = self.forward_debias_layers(nn_bias1)

        # Process through ranking layers
        nn_ranking = (nn_bias0 - nn_bias1) / 2.
        self.nn_ranking = self.forward_ranking_acitvation(nn_ranking)
        # AUX Ranker
        input_for_aux = in_0 / 2.
        self.nn_ranking_cls = self.forward_ranking_acitvation(input_for_aux, reuse=True)

        return nn_bias0, nn_bias1, nn_ranking, extracted_features
    
    def forward_extracted_features(self, x):
        for layer in self.feature_layers:
            x = self.feature_activation(layer(x))
        return x

    
    def forward_debias_layers(self, x):
        for layer in self.bias_layers:
            x = self.feature_activation(layer(x))
        return x
    

    def forward_ranking_acitvation(self, x, reuse=False):
        return self.ranking_activation(self.ranking_layer(x))
        

    def compute_loss(self, nn_output, y0, nn_bias0, y_bias_0, nn_bias1, y_bias_1, gamma):
        loss = nn.CrossEntropyLoss()
        #ranking_loss = loss(nn_output, y)
        bias0_loss = loss(nn_bias0, y_bias_0)
        bias1_loss = loss(nn_bias1, y_bias_1)
        ranking_loss = (y0 - nn_output) ** 2
        fairness_loss = gamma * (bias0_loss + bias1_loss)
        return torch.mean(ranking_loss), torch.mean(fairness_loss)
        
       
    
    
    def _get_feed_dict_queries(self, x, y, y_bias, samples, around=30):
        pairs_dict = super()._get_feed_dict_queries(x, y, y_bias, samples)

        return {self.x0: pairs_dict['x0'], self.x1: pairs_dict['x1'],
                self.y_bias_0: pairs_dict['y_bias_0'], self.y_bias_1: pairs_dict['y_bias_1']}
    

    @staticmethod
    def save(estimator, path):
        raise NotImplementedError
    

    @staticmethod
    def load_ranker(path):
        raise NotImplementedError
    
