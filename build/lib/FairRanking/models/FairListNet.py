import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from FairRanking.models.BaseDirectRanker import BaseDirectRanker


class FairListNet(BaseDirectRanker):
    """
    Tensorflow implementation of https://arxiv.org/pdf/1805.08716.pdf
    Inspired by: https://github.com/MilkaLichtblau/DELTR-Experiments

    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param activation: tf function for the feature part of the net
    :param kernel_initializer: tf kernel_initializer
    :param start_batch_size: cost function of FairListNet
    :param min_doc: min size of docs in query if a list is given
    :param end_batch_size: max size of docs in query if a list is given
    :param start_len_qid: start size of the queries/batch
    :param end_len_qid: end size of the queries/batch
    :param learning_rate: learning rate for the optimizer
    :param max_steps: total training steps
    :param learning_rate_step_size: factor for increasing the learning rate
    :param learning_rate_decay_factor: factor for increasing the learning rate
    :param optimizer: tf optimizer object
    :param print_step: for which step the script should print out the cost for the current batch
    :param weight_regularization: float for weight regularization
    :param dropout: float amount of dropout
    :param input_dropout: float amount of input dropout
    :param name: name of the object
    :param num_features: number of input features
    :param protected_feature_deltr: column name of the protected attribute (index after query and document id)
    :param gamma_deltr: value of the gamma parameter
    :param iterations_deltr: number of iterations the training should run
    :param standardize_deltr: let's apply standardization to the features
    :param random_seed: random seed
    """

    def __init__(self,
                 hidden_layers=[10],
                 feature_activation=tf.nn.tanh,
                 ranking_activation=tf.nn.tanh,
                 kernel_initializer=tf.random_normal_initializer(),
                 start_batch_size=100,
                 end_batch_size=3000,
                 start_qids=10,
                 end_qids=100,
                 learning_rate=0.01,
                 max_steps=3000,
                 dataset=None,
                 max_queries=50,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.944,
                 optimizer=tf.train.AdamOptimizer,
                 print_step=0,
                 name="FairListNet",
                 gamma=1,
                 feature_bias=True,
                 noise_module=False,
                 noise_type='sigmoid_full',
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1.,
                 whiteout_lambda=1.,
                 num_features=0,
                 num_fair_classes=0,
                 random_seed=42,
                 save_dir=None
                 ):
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

        self.feature_layers = nn.ModuleList()
        prev_nodes = num_features
        for num_nodes in self.hidden_layers:
            # TODO: for listNet? rewrite?
            layer = nn.Linear(prev_nodes, num_nodes)
            self.feature_layers.append(layer)

        

        



    def forward(self, x):
        if self.noise_module:
            in_x = self.create_noise_module_list_net(self.x)
        else:
            in_x = self.x