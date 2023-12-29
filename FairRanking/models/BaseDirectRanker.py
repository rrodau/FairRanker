import numpy as np

import torch
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


def build_pairs1(x, y, s, samples_per_pair):
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
    for i in range(len(keys)):
        for j in range(len(keys)):
            if counts[i] == 0 or counts[j] == 0:
                continue  # skip if no samples for a key
            if keys[i] == keys[j]:
                continue
            n_samples = min(samples_per_pair, counts[i], counts[j])
            indices0 = np.random.randint(0, counts[i], n_samples)
            indices1 = np.random.randint(0, counts[j], n_samples)

            querys0 = np.where(y == keys[i])[0]
            querys1 = np.where(y == keys[j])[0]
            #indices0 = np.random.choice(np.where(y == keys[i])[0], n_samples, replace=True)
            #indices1 = np.random.choice(np.where(y == keys[j])[0], n_samples, replace=True)

            x0.extend(x[querys0][indices0][:, :len(x)])
            x1.extend(x[querys1][indices1][:, :len(x)])
            s0.extend(s[querys0][indices0][:, :len(x)])
            s1.extend(s[querys1][indices1][:, :len(x)])
            #x0.extend(x[indices0][:, :len(x)])
            #x1.extend(x[indices1][:, :len(x)])
            #s0.extend(s[indices0][:, :len(x)])
            #s1.extend(s[indices1][:, :len(x)])
            # Determine the ranking relationship
            if keys[i] == keys[j]:
                y_train.extend(np.zeros(n_samples))  # Equal ranking
            elif keys[i] > keys[j]:
                y_train.extend(np.ones(n_samples))   # x0 ranked higher
            else:
                y_train.extend(-np.ones(n_samples))  # x1 ranked higher

    x0 = torch.tensor(x0, dtype=torch.float32)
    x1 = torch.tensor(x1, dtype=torch.float32)
    s0 = torch.tensor(s0, dtype=torch.float32)
    s1 = torch.tensor(s1, dtype=torch.float32)
    y_train = torch.tensor(np.array([y_train]).transpose(), dtype=torch.float32)

    return x0, x1, y_train, s0, s1


def build_pairs2(x, y, s, samples_per_pair):
    x0 = []
    x1 = []
    s0 = []
    s1 = []
    y_train = []
    np.random.seed(42)

    # Determine the indices for male and female
    male_indices = np.where(s[:, 0] == 1)[0]
    female_indices = np.where(s[:, 1] == 1)[0]

    # Find the smaller class size
    #min_class_size = min(len(male_indices), len(female_indices))
    #n_samples = min(samples_per_pair, min_class_size)
    n_samples = samples_per_pair
    # Generate pairs with balanced representation
    for _ in range(n_samples):
        # Randomly select male and female indices
        male_idx = np.random.choice(male_indices, 1, replace=True)[0]
        female_idx = np.random.choice(female_indices, 1, replace=True)[0]

        # Add pairs to the lists
        x0.append(x[male_idx])
        x1.append(x[female_idx])
        s0.append(s[male_idx])
        s1.append(s[female_idx])

        # Determine y_train based on y values
        y_diff = y[male_idx] - y[female_idx]
        y_train.append(np.sign(y_diff))

    # Combine and shuffle the pairs
    combined = list(zip(x0, x1, s0, s1, y_train))
    np.random.shuffle(combined)
    x0[:], x1[:], s0[:], s1[:], y_train[:] = zip(*combined)

    # Convert lists to tensors
    x0 = torch.tensor(x0, dtype=torch.float32)
    x1 = torch.tensor(x1, dtype=torch.float32)
    s0 = torch.tensor(s0, dtype=torch.float32)
    s1 = torch.tensor(s1, dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

    return x0, x1, s0, s1, y_train


def build_pairs(x, y, y_bias, samples):
    x0 = []
    x1 = []
    y_train = []
    y0_bias = []
    y1_bias = []
    y_bias_train = []

    keys, counts = np.unique(y, return_counts=True)
    sort_ids = np.argsort(keys)
    keys = keys[sort_ids]
    for i in range(len(keys) - 1):
        indices0 = np.random.randint(0, counts[i + 1], samples)
        indices1 = np.random.randint(0, counts[i], samples)
        querys0 = np.where(y == keys[i + 1])[0]
        querys1 = np.where(y == keys[i])[0]
        x0.extend(x[querys0][indices0])
        x1.extend(x[querys1][indices1])
        y_train.extend((keys[i + 1] - keys[i]) * np.ones(samples))
        y0_bias.extend(y_bias[querys0][indices0])
        y1_bias.extend(y_bias[querys1][indices1])
        tmp_0 = np.array(y_bias[querys0][indices0])
        tmp_1 = np.array(y_bias[querys1][indices1])
        y_bias_train.extend(tmp_0 - tmp_1)

    x0 = np.array(x0)
    x1 = np.array(x1)
    y_bias_train = np.array(y_bias_train)[:, 0]
    y_train = np.array([y_train]).transpose()
    y0_bias = np.array(y0_bias)
    y1_bias = np.array(y1_bias)
    y_bias_train = np.expand_dims(y_bias_train, axis=1)
    _, y_bias_counts = np.unique(y_bias_train, return_counts=True)
    if np.shape(y_bias_counts)[0] == 1:
        num_nonzeros = 1
    else:
        num_nonzeros = y_bias_counts[1]
    adj_sym_loss_term = len(x0) / num_nonzeros

    # Convert numpy arrays to PyTorch tensors
    x0_tensor = torch.tensor(x0, dtype=torch.float32)
    x1_tensor = torch.tensor(x1, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y0_bias_tensor = torch.tensor(y0_bias, dtype=torch.float32)
    y1_bias_tensor = torch.tensor(y1_bias, dtype=torch.float32)

    return x0_tensor, x1_tensor, y_train_tensor, y0_bias_tensor, y1_bias_tensor

# Example usage
# x, y, y_bias would be your input data
# x0_tensor, x1_tensor, y_train_tensor, y0_bias_tensor, y1_bias_tensor will be the outputs


# Needs to got to BaseDatasets
def convert_data_to_tensors(data):
    (X_train, s_train, y_train), (X_val, s_val, y_val), (X_test, s_test, y_test) = data.get_data()
    X_train0, X_train1, y_train, s_train0, s_train1 = build_pairs1(X_train, y_train, s_train, X_train.shape[0])
    X_val0, X_val1, y_val, s_val0, s_val1 = build_pairs1(X_val, y_val, s_val, X_val.shape[0])
    X_test0, X_test1, y_test, s_test0, s_test1= build_pairs1(X_test, y_test, s_test, X_test.shape[0])
    return X_train0, X_train1, s_train0, s_train1, y_train, X_val0, X_val1, s_val0, s_val1, y_val, X_test0, X_test1, s_test0, s_test1, y_test