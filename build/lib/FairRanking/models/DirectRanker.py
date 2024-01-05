import torch
import torch.nn as nn
import numpy as np

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
                 num_features=0,
                 random_seed=42,
                 start_batch_size=100,
                 end_batch_size=500,
                 end_qids=300,
                 start_qids=10,
                 name="DirectRanker",
                 ):
        if random_seed:
            torch.manual_seed(random_seed)
        super().__init__()
        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.kernel_initializer = kernel_initializer
        self.end_qids = end_qids
        self.start_qids = start_qids
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
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

        #self.ranking_layer_cls = nn.Linear(prev_nodes, 1, bias=False)


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
        mm = self.ranking_layer(self.ranking_activation(mm))
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
        return mm0 - mm1 / 2
    

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_extracted_features(x)
        res = self.ranking_activation(self.ranking_layer(x / 2.))
        return 0.5 * (1 + res)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_extracted_features(x)
        res =  self.ranking_activation(self.ranking_layer(x / 2.))
        return (res > 0).int()


    def get_feed_dict(self, x, y, y_bias, samples):
        # TODO: Optimize
        """
        Prepares a dictionary for training with pairs of data points from adjacent classes.

        Arguments:
        x (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        y_bias (torch.Tensor): Bias labels.
        samples (int): Number of samples to draw from each class.

        Returns:
        dict: A dictionary containing pairs of data points, training labels, bias labels, and an adjustment term for loss.
        """
        x0, x1, y_train, y0_bias, y1_bias, y_bias_train = [], [], [], [], [], []

        keys, counts = torch.unique(y, return_counts=True)
        # go through each unique class
        for i in range(len(keys) - 1):
            # random indices for current and next class
            indices0 = torch.randint(0, counts[i + 1], (samples,))
            indices1 = torch.randint(0, counts[i], (samples,))

            # get the indices where y equals the current and next class
            querys0 = torch.where(y == keys[i + 1])[0]
            querys1 = torch.where(y == keys[i])[0]

            # Append pairs of data, labels and bias labels
            # Isn't there a possible IndexOutOfBounds Error?
            x0.append(x[querys0][indices0])
            x1.append(x[querys1][indices1])
            y_train.append((keys[i + 1] - keys[i]).repeat(samples))
            y0_bias.append(y_bias[querys0][indices0])
            y1_bias.append(y_bias[querys1][indices1])
            y_bias_train.append((y_bias[querys0][indices0] - y_bias[querys1][indices1]).squeeze())

        # Concatenate lists into tensors
        x0 = torch.cat(x0)
        x1 = torch.cat(x1)
        y_train = torch.cat(y_train).unsqueeze(1)
        y0_bias = torch.cat(y0_bias)
        y1_bias = torch.cat(y1_bias)
        y_bias_train = torch.cat(y_bias_train).unsqueeze(1)

        # Calculate an adjustment term for symmetric loss
        y_bias_counts = torch.unique(y_bias_train, return_counts=True)[1]
        num_nonzeros = y_bias_counts[1] if len(y_bias_counts) > 1 else 1
        adj_sym_loss_term = len(x0) / num_nonzeros

        return {'x0': x0, 'x1': x1, 'y_train': y_train,
                'y_bias_0': y0_bias, 'y_bias_1': y1_bias,
                'y_bias': y_bias_train, 'adj_loss_term': adj_sym_loss_term}
    

    def get_feed_dict_queries(self, x, y, y_bias, samples, around=30):
        """
        :param current_batch:
        :param around:
        """
        x0 = []
        x1 = []
        y_train = []
        y0_bias = []
        y1_bias = []

        keys, counts = torch.unique(y, return_counts=True, sorted=True)
        indices0 = torch.randint(0, len(keys), (samples,))
        diff_indices1 = torch.randint(-around, around, (samples,))
        indices1 = []
        for j in range(len(indices0)):
            if diff_indices1[j] == 0:
                diff_indices1[j] = 1
            tmp_idx = (indices0[j] + diff_indices1[j]) % len(keys)
            if tmp_idx > indices0[j]:
                indices1.append(indices0[j])
                indices0[j] = tmp_idx
            else:
                indices1.append(tmp_idx)
            assert indices0[j] >= indices1[j]
        x0 = x[indices0]
        x1 = x[indices1]
        print(len(x0[0]))

        y_train.extend(1 * np.ones(samples))
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        y0_bias = y_bias[indices0]
        y1_bias = y_bias[indices1]

        return {'x0': x0, 'x1': x1, 'y_train': y_train,
                'y_bias_0': y0_bias, 'y_bias_1': y1_bias}