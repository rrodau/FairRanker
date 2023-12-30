import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from FairRanking.models.BaseDirectRanker import BaseDirectRanker
from FairRanking.models.flip_gradient import flipGradient


class DebiasClassifier(BaseDirectRanker):

    def __init__(self,
                 hidden_layers=[10, 5],
                 bias_layers=[50, 20, 2],
                 feature_activation=nn.Sigmoid(),
                 ranking_activation=nn.Sigmoid(),
                 feature_bias=True,
                 kernel_initializer=nn.init.normal_,
                 random_seed=42,
                 dataset=None,
                 name="DebiasClassifier",
                 gamma=1,
                 noise_module=False,
                 noise_type="sigmoid_full",
                 whiteout=False,
                 uniform_noise=0,
                 whiteout_gamma=1,
                 whiteout_lambda=1,
                 num_features=0,
                 num_fair_classes=0,
                 save_dir=None,
                 num_relevance_classes=0):
        super().__init__(hidden_layers=hidden_layers,
                         feature_activation=feature_activation, ranking_activation=ranking_activation,
                         feature_bias=feature_bias, kernel_initializer=kernel_initializer,
                         random_seed=random_seed, name=name, gamma=gamma,
                         noise_module=noise_module, noise_type=noise_type, whiteout=whiteout,
                         uniform_noise=uniform_noise,
                         whiteout_gamma=whiteout_gamma, whiteout_lambda=whiteout_lambda, num_features=num_features,
                         num_fair_classes=num_fair_classes, save_dir=save_dir)
        
        self.bias_layers = bias_layers
        self.num_relevance_classes = num_relevance_classes

        self.feature_layers = nn.ModuleList()
        prev_neurons = self.num_features

        # Creating hidden layers for extracted features
        for num_neurons in self.hidden_layers:
            layer = nn.Linear(prev_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.feature_layers.append(layer)
            prev_neurons = num_neurons

        # Creating bias Layers
        prev_bias_neurons = prev_neurons
        self.debias_layers = nn.ModuleList()  
        for num_neurons in self.bias_layers:
            layer = nn.Linear(prev_bias_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.debias_layers.append(layer)
            prev_bias_neurons = num_neurons

        # Additional Layers for the Main Network which go up to the last of the bias layers
        self.additional_main_layers = nn.ModuleList()
        for num_neurons in self.bias_layers[:-1]:
            layer = nn.Linear(prev_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.additional_main_layers.append(layer)
        
        # output of the relevance classes in the main network for classification
        self.output_layer = nn.Linear(prev_neurons, self.num_relevance_classes)

    
    def forward_extracted_features(self, x):
        """
        Helper function for the forward pass. It passes the input into the layers for the
        extracted feautres

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the layers which
                            extracts the features
        
        Returns:
        - torch.Tensor: The extracted features of the document
        """
        for layer in self.feature_layers:
            x = self.feature_activation(layer(x))
        return x
    

    def forward_nn(self, x):
        """
        Helper function for the forward pass. It passes the extracted features
        into the last layers and the into the output layers

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the last
                            layers for the main classification
        
        Returns:
        - torch.Tensor: The logits for the main classification
        """
        x = self.forward_extracted_features(x)
        for layer in self.additional_main_layers:
            x = self.feature_activation(layer(x))
        x = self.output_layer(x)
        return x
    

    def forward_debias(self, x):
        """
        Helper function for the forward pass of the bias layer. 
        It passes the extracted features into the debias layers

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the layers in the
                            the bias layers
        
        Returns:
        - torch.Tensor: The logits for classification of the sensible attribute
        """
        x = self.forward_extracted_features(x)
        # here is the gradient flip
        for layer in self.debias_layers:
            x = self.feature_activation(layer(x))
        return x
