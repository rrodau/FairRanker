import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from FairRanking.models.BaseDirectRanker import BaseDirectRanker
from FairRanking.models.flip_gradient import CustomLayer


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
                 start_batch_size=100,
                 end_batch_size=500,
                 end_qids=300,
                 start_qids=10,
                 name="DebiasClassifier",
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
                         random_seed=random_seed, start_batch_size=start_batch_size,
                         end_batch_size=end_batch_size, start_qids=start_qids, end_qids=end_qids, name=name,
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
            prev_neurons = num_neurons
        
        # output of the relevance classes in the main network for classification
        self.output_layer = nn.Linear(prev_neurons, self.num_relevance_classes)
 
        self.flip_gradient = CustomLayer()


    def forward_extracted_features(self, x):
        """
        Helper function for the forward pass. It passes the input into the hidden layers to get the
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
        Helper function for the forward pass. It passes input through the main
        network

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
        A function which passes the input through the hidden layers to get the
        extracted features and then pass these through the debias layers

        Parameters:
        - x (torch.Tensor): An input document which will be passed through the layers in the
                            the bias layers
        
        Returns:
        - torch.Tensor: The logits for classification of the sensible attribute
        """
        x = self.forward_extracted_features(x)
        x = self.flip_gradient(x)
        for layer in self.debias_layers:
            x = self.feature_activation(layer(x))
        return x


    def predict_proba(self, x):
        x = self.forward_nn(x)
        return nn.Softmax(dim=1)(x)
    

    def predict(self, x):
        x = self.forward_nn(x)
        return torch.argmax(x, dim=1)
    
    
    def get_feed_dict(self, x, y, y_bias, samples):
        idx = torch.randperm(len(x))[:samples]
        x_batch = x[idx]
        y_batch = one_hot_convert(y[idx], self.num_relevance_classes)
        y_bias_batch = y_bias[idx]
        return {'x': x_batch, 'y': y_batch, 's': y_bias_batch}


def one_hot_convert2(y, num_classes):
    y_one_hot = torch.zeros((y.size(0), num_classes), dtype=torch.float32)
    y_one_hot.scatter_(1, y.unsqueeze(1).long() - 1, 1)
    return y_one_hot

def one_hot_convert(y, num_classes):
    y.detach().numpy()
    arr = np.zeros((len(y), num_classes))
    for i, yi in enumerate(y):
        arr[i, int(yi) - 1] = 1
    return torch.tensor(arr, dtype=torch.float32)