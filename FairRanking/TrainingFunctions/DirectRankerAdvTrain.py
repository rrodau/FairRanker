import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from FairRanking.helpers import nDCG_cls, disparate_impact, calc_accuracy, rND_torch, auc_estimator, group_pairwise_accuracy, calc_rnd, disparity_loss
from FairRanking.models.DirectRankerAdv import DirectRankerAdv
from torch.optim.lr_scheduler import StepLR


def main_phase(model, X_train0, X_train1, y_train, optimizer, loss_fn):
    y_pred_train = model(X_train0, X_train1)
    main_loss = loss_fn(y_train, y_pred_train)
    main_loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()

def adversarial_phase(model, X_train0, X_train1, s_train0, s_train1, optimizer, loss_fn, loss_threshold):
    sensitive_pred0, sensitive_pred1 = model.forward_2(X_train0, X_train1)
    sensitive_loss = loss_fn(torch.cat((sensitive_pred0, sensitive_pred1), dim=0), torch.cat((s_train0, s_train1), dim=0))
    sensitive_loss.backward()

    is_loss_below_threshold = sensitive_loss.item() < loss_threshold
    for name, param in model.named_parameters():
        if 'debias' in name:
            continue
        else:
            if param.grad is not None:
                if is_loss_below_threshold:
                    param.grad.neg_()
                else:
                    param.grad.zero_()

    optimizer.step()
    optimizer.zero_grad()
    return sensitive_loss
 
 
def only_adversarial(model, X_train0, X_train1, s_train0, s_train1, optimizer, loss_fn):
    sensitive_pred0, sensitive_pred1 = model.forward_2(X_train0, X_train1)
    sensitive_loss = loss_fn(torch.cat((sensitive_pred0, sensitive_pred1), dim=0), torch.cat((s_train0, s_train1), dim=0))
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
          threshold = 0.4,
          n_epochs = 1000,
          use_validation = False,
          random_seed=42,
          path=None):
    if path:
        with open(f'{path}results_extra_sensitive_training.csv', 'w') as file:
            file.write(f'Schedule,SensitiveLoss\n')
    if path:
        with open(f'{path}results.csv', 'w') as file:
            file.write(f'AUC,Accuracy,rND,NDCG,GPA,SensitiveLoss,Schedule\n')
    X_train0, X_train1, y_train, s_train0, s_train1 = data[0]
    X_test0, X_test1, y_test, s_test0, s_test1 = data[1]
    if use_validation:
        X_val0, X_val1, y_val, s_val0, s_val1 = data[2]
    n_epochs = int(n_epochs / schedule[0])
    lr_decay = lr_decay
    main_optimizer = main_optimizer(model.parameters(), lr=main_lr)
    adv_optimizer = adv_optimizer(model.parameters(), lr=adv_lr)
    opt_scheduler = StepLR(main_optimizer, step_size=lr_step, gamma=lr_decay)
    adv_scheduler = StepLR(adv_optimizer, step_size=lr_step, gamma=lr_decay)
    loss_fn = nn.MSELoss(reduction='mean')
    sensitive_loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        model.train()
        for _ in range(schedule[0]):
            main_phase(model, X_train0, X_train1, y_train, main_optimizer, loss_fn)
        for _ in range(schedule[1]):
            sensitive_loss = adversarial_phase(model, X_train0, X_train1, s_train0, s_train1, adv_optimizer, sensitive_loss_fn, threshold)
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test0, X_test1)
        test_loss = loss_fn(y_test, y_test_pred)
        test_acc = calc_accuracy(y_test_pred, y_test)
        di_test_score = disparate_impact(y_test_pred, s_test0, s_test1)
        print(f'Test Loss: {test_loss.item():.4f}\t Test Accuracy: {test_acc.item():.4f}\t DI: {di_test_score:.4f}')
        auc_test = auc_estimator(y_test_pred, y_test)
        ndcg_test = nDCG_cls(model, X_test0, X_test1, y_test, esti=False)
        gpa_test = group_pairwise_accuracy(y_test_pred, y_test, s_test0)
        sensitive_pred0, sensitive_pred1 = model.forward_2(X_test0, X_test1)
        sensitive_loss = sensitive_loss_fn(torch.cat((sensitive_pred0, sensitive_pred1), dim=0), torch.cat((s_test0, s_test1), dim=0))
        rnd_arr = []
        for _ in range(10):
            rnd_arr.append(calc_rnd(model, X_test0, X_test1, s_test0, s_test1))
        rnd_test = np.mean(rnd_arr)
        if path:
            with open(f'{path}results.csv', 'a+') as file:
                file.write(f'{auc_test},{test_acc.item()},{rnd_test},{ndcg_test},{gpa_test},{sensitive_loss},{str(schedule).replace(",",";")}\n')
    for _ in range(n_epochs):
        for _ in range(schedule[1]):
            model.train()
            sensitive_loss = only_adversarial(model, X_train0, X_train1, s_train0, s_train1, adv_optimizer, sensitive_loss_fn)
    model.eval()
    with torch.no_grad():
        sensitive_pred0, sensitive_pred1 = model.forward_2(X_test0, X_test1)
        test_sensitive_loss = sensitive_loss_fn(torch.cat((sensitive_pred0, sensitive_pred1), dim=0), torch.cat((s_test0, s_test1), dim=0))
        if path:
            with open(f'{path}results_extra_sensitive_training.csv', 'a+') as file:
                file.write(f'{str(str(schedule).replace(",",";"))},{test_sensitive_loss}\n')
    print(f'Finished Schedule: {schedule}')