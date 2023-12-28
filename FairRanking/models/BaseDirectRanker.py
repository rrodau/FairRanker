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
        :param print_step: for which step the script should print out the cost for the current batch
    """

    def __init__(self,
                 hidden_layers=[10, 5],
                 feature_activation=nn.Tanh,
                 ranking_activation=nn.Tanh,
                 feature_bias=True,
                 kernel_initializer=nn.init.normal_,
                 random_seed=42,
                 name="DirectRanker",
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
        super().__init__()
        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.dataset = dataset
        self.kernel_initializer = kernel_initializer
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
        
def build_pairs(x, y, s, samples):
    """
    :param x:
    :param y:
    :param y_bias:
    :param samples:
    """
    x0 = []
    x1 = []
    s0 = []
    s1 = []
    y_train = []
    np.random.seed(42)
    keys, counts = np.unique(y, return_counts=True)
    sort_ids = np.argsort(keys)
    keys = keys[sort_ids]
    counts = counts[sort_ids]
    for i in range(len(keys) - 1):
        indices0 = np.random.randint(0, counts[i + 1], samples)
        indices1 = np.random.randint(0, counts[i], samples)
        querys0 = np.where(y == keys[i + 1])[0]
        querys1 = np.where(y == keys[i])[0]
        x0.extend(x[querys0][indices0][:, :len(x)])
        x1.extend(x[querys1][indices1][:, :len(x)])
        s0.extend(s[querys0][indices0][:, :len(x)])
        s1.extend(s[querys1][indices1][:, :len(x)])

        y_train.extend((keys[i + 1] - keys[i]) * np.ones(samples))

    x0 = torch.tensor(x0, dtype=torch.float32)
    x1 = torch.tensor(x1, dtype=torch.float32)
    s0 = torch.tensor(s0, dtype=torch.float32)
    s1 = torch.tensor(s1, dtype=torch.float32)
    y_train = torch.tensor(np.array([y_train]).transpose(), dtype=torch.float32)

    return x0, x1, s0, s1, y_train