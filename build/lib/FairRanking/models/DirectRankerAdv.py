import torch
import torch.nn as nn

from FairRanking.models.BaseDirectRanker import BaseDirectRanker
from FairRanking.models.flip_gradient import gradient_reversal_layer

class DirectRankerAdv(BaseDirectRanker):
    """
    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param feature_activation: tf function for the feature part of the net
    :param ranking_activation: tf function for the ranking part of the net
    :param feature_bias: boolean value if the feature part should contain a bias
    :param kernel_initializer: tf kernel_initializer
    :param gamma: value how important the fair loss is
    """
    
    def __init__(self,
                 hidden_layers=[10, 5],
                 bias_layers=[32, 16],
                 feature_activation=nn.Tanh(),
                 ranking_activation=nn.Tanh(),
                 feature_bias=True,
                 kernel_initializer=nn.init.normal_,
                 random_seed=42,
                 name="DirectRankerAdv",
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
                    random_seed=random_seed, name=name, noise_module=noise_module, 
                    noise_type=noise_type, whiteout=whiteout, uniform_noise=uniform_noise,
                    whiteout_gamma=whiteout_gamma, whiteout_lambda=whiteout_lambda, 
                    num_features=num_features,
                    num_fair_classes=num_fair_classes, save_dir=save_dir)
        
        # Feature Layers
        self.feature_layers = nn.ModuleList()
        prev_neurons_extracted = num_features
        for num_neurons in hidden_layers:
            # Create Layer
            layer = nn.Linear(prev_neurons_extracted, num_neurons)
            # Initialize weights
            self.kernel_initializer(layer.weight)
            self.feature_layers.append(layer)
            prev_neurons_extracted = num_neurons
        prev_neurons = prev_neurons_extracted

        # Debias Layers
        self.debias_layers = nn.ModuleList()
        for num_neurons in bias_layers:
            # Create Layer
            layer = nn.Linear(prev_neurons, num_neurons)
            # Initialize weights
            self.kernel_initializer(layer.weight)
            self.debias_layers.append(layer)
            prev_neurons = num_neurons
        self.debias_layers.append(nn.Linear(prev_neurons, 2))
        # Ranking Layers
        # Output from the Feature Layers goes into the Ranking Layer 
        self.ranking_layer = nn.Linear(prev_neurons_extracted, 1, bias=False)
        

#TODO: Do I realy need to recompute the extracted featres in the bias forward pass?
    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        The main forward function of the DirectRankerAdv computes the rank score for
        the documents provided

        Parameters:
        - x0 (torch.Tensor): A document which needs to be ranked in relation to another document x1
        - x1 (torch.Tesnor): A document which needs to be ranked in relation to another document x0

        Returns:
        - torch.Tensor: Ranking score given to the paired documents. Values \in (0,-1] 
                        mean the document x1 is ranked higher whereas values \in (0, 1]
                        mean the document x0 is ranked higher.
        """

        # Apply noise module if enabled
        if self.noise_module:
            in_0, in_1 = self.create_noise_module(x0, x1)
        else:
            in_0, in_1 = x0, x1

        # Process through Feature Layers
        in_0 = self.forward_extracted_features(in_0)
        in_1 = self.forward_extracted_features(in_1)

        # Process through Ranking Layers
        nn_ranking = in_0 - in_1
        nn_ranking = self.forward_ranking_acitvation(nn_ranking)

        return nn_ranking
    

    def forward_2(self, x0: torch.Tensor, x1: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        The second forward function which is used for the adversarial network which tries to predict
        the sensible attribute. It reverses the Gradient into the Main Network during backpropagation

        Parameters:
        - x0 (torch.Tensor): A document which needs to be ranked in relation to another document x1
                             and which sensible attribute needs to be predicted
        - x1 (torch.Tesnor): A document which needs to be ranked in relation to another document x0
                             and which sensible attribute needs to be predicted

        Returns:
        - (torch.Tensor, torch.Tensor): The sensible attribute the model predicted for each document
        """
        if self.noise_module:
            in_0, in_1 = self.create_noise_module(x0, x1)
        else:
            in_0, in_1 = x0, x1

        # Process through Feature Layers
        in_0 = self.forward_extracted_features(in_0)
        in_1 = self.forward_extracted_features(in_1)

        # Apply gradient reversal for bias handling
        #in_0 = gradient_reversal_layer(in_0)
        #in_1 = gradient_reversal_layer(in_1)

        # Process through the Debias Layers which predict the sensible attribute
        nn_pred_sensitive_0 = self.forward_debias_layers(in_0)
        nn_pred_sensitive_1 = self.forward_debias_layers(in_1)

        return nn_pred_sensitive_0, nn_pred_sensitive_1


    def forward_extracted_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the forward pass. It passes the input into the layers for the
        extracted feautres

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the layers in the
                            the main network except the ranking layer
        
        Returns:
        - torch.Tensor: The extracted features of the document
        """
        for layer in self.feature_layers:
            x = self.feature_activation(layer(x))
        return x

    
    def forward_debias_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function for the second forward pass. It passes the input into the layers for the
        predicted sensible attributes

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the layers in the
                            the adversarial network
        
        Returns:
        - torch.Tensor: The prediction of the sensible attribute
        """
        for layer in self.debias_layers[:-1]:
            x = self.feature_activation(layer(x))
        return self.debias_layers[-1](x)
    

    def forward_ranking_acitvation(self, x: torch.Tensor, reuse=False) -> torch.Tensor:
        """
        Helper function for the ranking output. It passes the extracted features
        into a ranking activation

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the ranking 
                            layer in the the main network
        
        Returns:
        - torch.Tensor: Ranking score given to the paired documents. Values \in (0,-1] 
                        mean the document x1 is ranked higher whereas values \in (0, 1]
                        mean the document x0 is ranked higher.
        """
        return self.ranking_activation(self.ranking_layer(x))
        

    @staticmethod
    def save(estimator, path):
        raise NotImplementedError
    

    @staticmethod
    def load_ranker(path):
        raise NotImplementedError
