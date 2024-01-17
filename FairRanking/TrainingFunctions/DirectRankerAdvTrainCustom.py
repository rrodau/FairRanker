import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
from FairRanking.helpers import nDCG_cls, disparate_impact, calc_accuracy, rND_torch, auc_estimator, group_pairwise_accuracy, calc_rnd, disparity_loss, calc_sens_loss
from FairRanking.models.DirectRankerAdv import DirectRankerAdv
from torch.optim.lr_scheduler import StepLR
from time import time

def main_phase(model, X_train0, X_train1, y_train, optimizer, loss_fn):
    y_pred_train = model(X_train0, X_train1)
    main_loss = loss_fn(y_train, y_pred_train)
    main_loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    return main_loss.item()


def adversarial_phase(model, X_train0, X_train1, s_train0, s_train1, optimizer, loss_fn, loss_threshold):
    sensitive_pred0, sensitive_pred1 = model.forward_2(X_train0, X_train1)
    sensitive_loss = loss_fn(torch.cat((sensitive_pred0, sensitive_pred1), dim=0), torch.cat((s_train0, s_train1), dim=0))
    sensitive_loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return sensitive_loss.item()
 
 
def only_adversarial(model, X_train0, X_train1, s_train0, s_train1, optimizer, loss_fn):
    sensitive_pred0, sensitive_pred1 = model.forward_2(X_train0, X_train1)
    sensitive_loss = calc_sens_loss(sensitive_pred0, sensitive_pred1, s_train0, s_train1)#loss_fn(torch.cat((sensitive_pred0, sensitive_pred1), dim=0), torch.cat((s_train0, s_train1), dim=0))
    sensitive_loss.backward()
    for p in model.named_parameters():
        if 'debias' in p[0]:
            continue
        else:
            if p[1].grad is not None:
                p[1].grad = torch.zeros_like(p[1].grad, dtype=torch.float32)

    optimizer.step()
    optimizer.zero_grad()
    return sensitive_loss.item()


def train(model,
          data,
          main_optimizer = optim.Adam,
          adv_optimizer = optim.Adam,
          lr_decay = 0.994,
          lr_step = 500,
          schedule = [1,1],
          main_lr = 0.001,
          adv_lr = 0.1,
          threshold = 1.5,
          n_epochs = 1000,
          use_validation = False,
          random_seed=42,
          is_query_dataset=False,
          print_res=True,
          writer=None,
          change_step=0.05):
    X_train, y_train, s_train = data[0]
    X_test, y_test, s_test = data[1]
    if use_validation:
        X_val, y_val, s_val = data[2]
    n_epochs = int(n_epochs / schedule[0])
    #n_schedules = sum(schedule)
    #schedule_list = ['main'] * schedule[0] + ['adv'] * schedule[1]
    lr_decay = lr_decay
    main_optimizer = main_optimizer(model.parameters(), lr=main_lr)
    adv_optimizer = adv_optimizer(model.parameters(), lr=adv_lr)
    opt_scheduler = StepLR(main_optimizer, step_size=lr_step, gamma=lr_decay)
    adv_scheduler = StepLR(adv_optimizer, step_size=lr_step, gamma=lr_decay)
    loss_fn = nn.MSELoss(reduction='mean')
    sensitive_loss_fn = nn.CrossEntropyLoss()
    sample_factor = np.log(
            1.0 * model.end_batch_size / model.start_batch_size)
    model.train()
    prev_main_loss = None
    prev_adv_loss = None
    # Initialize momentum terms
    momentum_main = 0
    momentum_adv = 0
    alpha = 0.2  # momentum factor
    for epoch in range(n_epochs):
        num_samples = int(model.start_batch_size * np.exp(
                1.0 * sample_factor * epoch / n_epochs))
        if not is_query_dataset:
            feed_dict = model.get_feed_dict(X_train, y_train, s_train, num_samples)
        else:
            feed_dict = model.get_feed_dict_queries(X_train, y_train, s_train, num_samples)
        for _ in range(schedule[0]):
            loss = main_phase(model, feed_dict['x0'], feed_dict['x1'], feed_dict['y_train'], main_optimizer, loss_fn)
        for _ in range(schedule[1]):
            adv_loss = adversarial_phase(model, feed_dict['x0'], feed_dict['x1'], feed_dict['y_bias_0'], feed_dict['y_bias_1'], adv_optimizer, sensitive_loss_fn, threshold) 
        if prev_adv_loss is not None and prev_main_loss is not None:
            """diff_main = 1/loss*  (prev_main_loss - loss)
            diff_adv = 1/adv_loss * (prev_adv_loss - adv_loss)
            #if diff_main > diff_adv:
            if diff_main > abs(0.5-adv_loss):
                if schedule[0] > 1:
                    schedule[0] -= 1
                
                schedule[1] += 1 if schedule[1] < 4 else 0
            #elif diff_main < diff_adv:
            elif diff_main < abs(0.5-adv_loss):
                if schedule[1] > 1:
                    schedule[1] -= 1
                
                schedule[0] += 1 if schedule[0] < 4 else 0"""

            # Compute the current improvement
            current_diff_main = prev_main_loss**0.5 - loss**0.5
            current_diff_adv = prev_adv_loss - adv_loss
            
            # Update momentum terms
            momentum_main = alpha * current_diff_main + (1 - alpha) * momentum_main
            momentum_adv = alpha * current_diff_adv + (1 - alpha) * momentum_adv

            if epoch % int(n_epochs * change_step) == 0:
                # Determine adjustment
                if momentum_main - momentum_adv > 0:
                    # Main loss is improving faster, decrease main schedule, increase adversarial
                    if loss < 0.5:
                        schedule[0] = max(1, schedule[0] - 1)
                
                    if momentum_main/momentum_adv > threshold:
                        schedule[1] = min(schedule[1] + 1, 5)
                elif momentum_adv - momentum_main > 0:
                    # Adversarial loss is improving faster, increase main schedule, decrease adversarial
                    #if adv_loss < 0.5:
                    #    schedule[1] = max(1, schedule[1] - 1)
                    if momentum_adv/momentum_main > threshold:
                        schedule[0] = min(schedule[0] + 1, 5)



                
        prev_main_loss = loss
        prev_adv_loss = adv_loss

        if writer is not None:
            input = {
                'Main Loss': [{'CrossEntropyLoss': loss}, epoch],
                'Adversarial Loss': [{'BCE Loss': adv_loss}, epoch],
                'Scheduling': [{'Main Schedule': schedule[0], 'Adv Schedule': schedule[1]}, epoch]
            }
            writer.write(**input)
        if epoch % 100 == 0 and print_res:
            if schedule[1]:
                print(f"Loss: {loss}\t Sens Loss: {adv_loss}")
            else:
                print(f"Loss: {loss}")
    if writer is not None:
        del writer
    return model