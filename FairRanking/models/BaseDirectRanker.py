import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

class BaseDirectRanker(nn.Module):
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
                 feature_activation=nn.Tanh,
                 ranking_activation=nn.Tanh,
                 feature_bias=True,
                 kernel_initializer=nn.init.normal_,
                 start_batch_size=100,
                 end_batch_size=500,
                 learning_rate=0.01,
                 max_steps=3000,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.944,
                 optimizer=optim.Adam,
                 print_step=0,
                 start_qids=0,
                 end_qids=10,
                 random_seed=42,
                 name="DirectRanker",
                 gamma=1,
                 dataset=None,
                 noise_module=False,
                 noise_type='sigmoid_full',
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1.,
                 whiteout_lambda=1.,
                 num_features=0,
                 num_fair_classes=0,
                 save_dir=None):
        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.dataset = dataset
        self.kernel_initializer = kernel_initializer
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.learning_rate_step_size = learning_rate_step_size
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.optimizer = optimizer
        self.print_step = print_step
        self.end_qids = end_qids
        self.start_qids = start_qids
        self.gamma = gamma
        self.noise_module = noise_module
        self.noise_type = noise_type
        self.whiteout = whiteout
        assert not (self.whiteout and self.noise_module)
        self.whiteout_gamma = whiteout_gamma
        self.whiteout_lambda = whiteout_lambda
        self.uniform_noise = uniform_noise
        self.random_seed = random_seed
        self.name = name
        self.num_features = num_features
        self.num_fair_classes = num_fair_classes
        self.y_rankers = {'lr_y': {}, 'rf_y': {}}
        self.s_rankers = {'lr_s': {}, 'rf_s': {}}
        self.checkout_steps = []
        if save_dir is None:
            self.save_dir = '/tmp/{}.format(self.name)'
        else:
            self.save_dir = save_dir
        self.load_path = self.save_dir + '/saved_model_step_{}'


    def forward(x):
        raise NotImplementedError


    def close_session(self):
        raise NotImplementedError
    

    def create_feature_layers(self, input_0, input_1, hidden_size, name):

        nn0 = nn.Linear(input_0, hidden_size, bias=self.feature_bias)
        self.kernel_initalizer(nn0.weight)

        nn1 = nn.linear(input_1, hidden_size, bias=self.feature_bias)
        self.kernel_initalizer(nn1.weight)

        return nn0, nn1


    def create_feature_layers_list_net(self, input, hidden_size):
        nn = nn.Linear(input, hidden_size, bias=self.feature_bias)
        self.kernel_initalizer(nn.weight)
    
        return nn

    
    def create_debias_layers(self, input_0, input_1, num_units=20):
        nn_bias0 = nn.Linear(input_0, num_units)
        self.kernel_initalizer(nn.weight)

        nn_bias1 = nn.Linear(input_0, num_units)
        self.kernel_initalizer(nn.weight)

        return nn_bias0, nn_bias1


    # I guess most of the functionality should go into the forward pass
    # what is an auxillary ranker?
    def creatin_ranking_layers(self, input_0, input_1, return_aux_ranker=True):
        nn = (input_0 - input_1) / 2. # Why? put it in the forward function?

        nn = nn.Linear(nn, 1, bias=False)
        self.kernel_initalizer(nn.weight)

        if return_aux_ranker:
            nn_cls = nn.Linear(input_0 / 2.) # how to use reuse in pytorch and what does it do
        
        return nn, nn_cls


    
    def create_noise_module(self, input_0, input_1):
        alpha = nn.Parameter(torch.empty(self.num_features))
        w_beta = nn.Parameter(torch.empty(self.num_features))


        self.kernel_initializer(alpha)
        self.kernel_initializer(w_beta)
        # Create noise tensor
        if self.uniform_noise == 1:
            noise = np.random.uniform(low=-1, high=1, size=self.num_features)
        else:
            noise = np.random.normal(size=self.num_features)           
        beta = torch.tensor(noise) * w_beta

        if self.noise_type == 'default':
            in_noise_0 = input_0 * alpha + beta
            in_noise_1 = input_1 * alpha + beta
        elif self.noise_type == 'sigmoid_full':
            in_noise_0 = torch.sigmoid(input_0 * alpha + beta)
            in_noise_1 = torch.sigmoid(input_1 * alpha + beta)
        elif self.noise_type == 'sigmoid_sep':
            in_noise_0 = torch.sigmoid(input_0 * alpha) + torch.sigmoid(beta)
            in_noise_1 = torch.sigmoid(input_1 * alpha) + torch.sigmoid(beta)
        elif self.noise_type == 'sigmoid_sep_2':
            in_noise_0 = (torch.sigmoid(input_0 * alpha) + torch.sigmoid(beta)) / 2
            in_noise_1 = (torch.sigmoid(input_1 * alpha) + torch.sigmoid(beta)) / 2

        return in_noise_0, in_noise_1


    def create_noise_module_list_net(self, input):
        alpha = nn.Parameter(torch.empty(self.num_features))
        w_beta = nn.Parameter(torch.empty(self.num_features))


        self.kernel_initializer(alpha)
        self.kernel_initializer(w_beta)
        # Create noise tensor
        if self.uniform_noise == 1:
            noise = np.random.uniform(low=-1, high=1, size=self.num_features)
        else:
            noise = np.random.normal(size=self.num_features)           
        beta = torch.tensor(noise) * w_beta

        if self.noise_type == 'default':
            in_noise_0 = input * alpha + beta
        elif self.noise_type == 'sigmoid_full':
            in_noise = torch.sigmoid(input * alpha + beta)
        elif self.noise_type == 'sigmoid_sep':
            in_noise = torch.sigmoid(input * alpha) + torch.sigmoid(beta)
        elif self.noise_type == 'sigmoid_sep2':
            in_noise = (torch.sigmoid(input * alpha) + torch.sigmoid(beta)) / 2
            
        return in_noise 
        