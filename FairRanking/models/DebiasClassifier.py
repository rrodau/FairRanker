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
                 feature_activation=nn.Sigmoid,
                 ranking_activation=nn.Sigmoid,
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
        # Creating hidden layers
        for i, num_neurons in enumerate(self.hidden_layers):
            layer = nn.Linear(prev_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.feature_layers.append(layer)
            prev_neurons = num_neurons

        self.bias_layers = nn.ModuleList()  
        for num_neurons in self.bias_layers:
            layer = nn.Linear(prev_bias_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.bias_layers.append(layer)
            prev_bias_neurons = num_neurons

        # additional layers
        self.additional_layers = nn.ModuleList()
        for num_neurons in self.bias_layers[:-1]:
            layer = nn.Linear(prev_neurons, num_neurons)
            self.kernel_initializer(layer.weight)
            self.additional_layers.append(layer)
        
        self.output_layer = nn.Linear(prev_neurons, self.num_relevance_classes)


    def forward(self, x):
        # Go through hidden layers
        for layer in self.layers:
            x = self.feature_activation(layer(x))
        
        # save the extracted features
        extracted_features = x.copy()

        # apply gradient reversal and go through bias layers
        x_bias = flipGradient(x)
        for i, layer in enumerate(self.bias_layers):
            activation = self.feature_activation if i != len(self.bias_layers) -1 else lambda x: x
            x_bias = activation(layer(x_bias))

        # go through additional layers
        for layer in self.bias_layers[:-1]:
            extracted_features = self.feature_activation(layer(extracted_features))
        
        # now through classification layers
        x_cls = self.output_layer(extracted_features)

        return x_bias, x_cls
    

def one_hot_convert(y, num_classes):
    arr = np.zeros((len(y), num_classes))
    for i, yi in enumerate(y):
        arr[i, int(yi)-1] = 1
    return arr

        

