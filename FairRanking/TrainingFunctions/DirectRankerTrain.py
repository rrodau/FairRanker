import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from time import time

def main_phase(model, X_train0, X_train1, y_train, optimizer, loss_fn):
    y_pred_train = model(X_train0, X_train1)
    main_loss = loss_fn(y_train, y_pred_train)
    main_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return main_loss.item()


def train(model,
          data,
          main_optimizer = optim.Adam,
          lr_decay = 0.994,
          lr_step = 500,
          main_lr = 0.001,
          n_epochs = 1000,
          use_validation = False,
          random_seed=42,
          is_query_dataset=False,
          print_res=True):
    torch.manual_seed(random_seed)

    X_train, y_train, s_train = data[0]
    X_test, y_test, s_test = data[1]
    if use_validation:
        X_val, y_val, s_val = data[2]
    lr_decay = lr_decay
    main_optimizer = main_optimizer(model.parameters(), lr=main_lr)
    opt_scheduler = StepLR(main_optimizer, step_size=lr_step, gamma=lr_decay)
    loss_fn = nn.MSELoss(reduction='mean')
    sample_factor = np.log(
            1.0 * model.end_batch_size / model.start_batch_size)
    model.train()
    for epoch in range(n_epochs):
        num_samples = int(model.start_batch_size * np.exp(
                1.0 * sample_factor * epoch / n_epochs))
        if not is_query_dataset:
            feed_dict = model.get_feed_dict(X_train, y_train, s_train, num_samples)
        else:
            feed_dict = model.get_feed_dict_queries(X_train, y_train, s_train, num_samples)
        loss = main_phase(model, feed_dict['x0'], feed_dict['x1'], feed_dict['y_train'], main_optimizer, loss_fn)
        
        if epoch % 100 == 0 and print_res:
            print(f"Loss: {loss}\t")
    return model