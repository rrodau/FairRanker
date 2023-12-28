
import numpy as np
import torch
import itertools
from sklearn.metrics import roc_auc_score


DATASET_LIST = ['compas', 'w3c', 'crime', 'german', 'trec', 'adult', 'synth', 'law-gender', 'law-race']

def disparate_impact(prediction, s0, s1):
    """
    specifficaly made for adult dataset
    """
    prediction = (prediction > 0).int()
    #prediction = prediction == 1
    # Convert predictions to boolean tensor and get indices where prediction is 1 or 0
    pred_1_indices = torch.where(prediction == 1)[0]
    pred_0_indices = torch.where(prediction == 0)[0]

    # Calculate positive cases for non-protected group
    positive_non_protected = torch.sum(s0[pred_1_indices, 1]) + torch.sum(s1[pred_0_indices, 1])

    # Calculate positive cases for protected group
    positive_protected = torch.sum(s0[pred_1_indices, 0]) + torch.sum(s1[pred_0_indices, 0])

    total_protected = torch.sum(s0[:,1] == 1)
    total_protected += torch.sum(s1[:,1] == 1)

    total_non_protected = torch.sum(s0[:,0] == 1)
    total_non_protected += torch.sum(s1[:,0] == 1)



    proportion_protected = positive_protected / total_protected
    proportion_non_protected = positive_non_protected / total_non_protected

    di_score = proportion_non_protected / proportion_protected

    return di_score.item()


def nDCG_cls(model, X0, X1, y, at=10, trec=False, reverse=True, k=1, m=1, esti=True):
    """
    Calculates the normalized Discounted Cumulative Gain (nDCG) for a given PyTorch model.

    This function evaluates the ranking quality of a PyTorch model using the nDCG metric. It supports both standard 
    and TREC-style nDCG calculations. The function can work with both binary and continuous model outputs, and 
    includes an option to handle multi-dimensional outputs.

    Parameters:
    - model: A PyTorch model that outputs rankings or scores.
    - X: The input tensor for the model, typically feature data.
    - y: The true relevance scores as a tensor, used as a reference for calculating the ideal ranking.
    - at (int, optional): The number of top items to consider in the nDCG calculation. Defaults to 10.
    - trec (bool, optional): Whether to use TREC-style nDCG calculation. Defaults to False.
    - reverse (bool, optional): Whether to sort the predictions in descending order. Defaults to True.
    - k (int, optional): A scaling factor used in the DCG calculation. Defaults to 1.
    - m (int, optional): Another scaling factor for the DCG calculation. Defaults to 1.
    - esti (bool, optional): Whether to treat the model's output as estimations (probabilities) and convert 
                             them to binary values. Defaults to True.

    Returns:
    - float: The calculated nDCG score.
    """

    model.eval()
    predictions = model(X0, X1)
    if esti:
        predictions = (predictions > 0).int()
    if len(predictions.size()) > 1 and predictions.size(1) > 1:
        predictions = torch.max(predictions, dim=1).values

    rand = torch.rand(predictions.shape)
    _, sorted_indices = torch.sort(torch.stack((predictions, rand), dim=1), dim=0, descending=reverse)
    sorted_list = predictions[sorted_indices[:, 1]].tolist()
    #sorted_list = sorted_list[0] # TODO: needs to be changed
    yref = sorted(y.tolist(), reverse=reverse)
    if trec:
        DCG = torch.zeros(1)
        IDCG = torch.zeros(1)
        max_value = max(sorted_list)
        for i in range(at):
            exp_dcg = sorted_list[i] + at - max_value
            exp_idcg = yref[i] + at - max_value
            exp_dcg = max(exp_dcg, 0)
            exp_idcg = max(exp_idcg, 0)
            DCG += (2 ** exp_dcg / (k / m) - 1) / torch.log2(torch.tensor(i + 2))
            IDCG += (2 ** exp_idcg - 1) / torch.log2(torch.tensor(i + 2))
        nDCG = (DCG / IDCG).item() if IDCG != 0 else 0
        return nDCG 
    else:
        DCG = 0
        IDCG = 0
        for i in range(min(at, len(sorted_list))):
            # TODO: Change extracting values form list
            #print('sorted List ', np.shape(sorted_list))
            #print('yref ', np.shape(yref))
            DCG += (2 ** sorted_list[i][0][0] - 1) / np.log2(i + 2)
            IDCG += (2 ** yref[i][0] - 1) / np.log2(i + 2)
        nDCG = DCG / IDCG
        return float(nDCG)
    

def auc_estimator(prediction, y):
    """
    Calculates the auc by using the sklearn roc_auc_score function
    """
    prediction = np.array(prediction.detach()).tolist()
    y = y.tolist()
    return roc_auc_score(y, prediction)


def group_pairwise_accuracy(estimator, X, y, y_bias):
    """Returns the group-dependent pairwise accuracies.

    Returns the group-dependent pairwise accuracies Acc_{G_i > G_j} for each pair
    of groups G_i \in {0, 1} and G_j \in {0, 1}.

    Args:
      prediction_diffs: NumPy array of shape (#num_pairs,) containing the
                        differences in scores for each ordered pair of examples.
      paired_groups: NumPy array of shape (#num_pairs, 2) containing the protected
                     groups for the better and worse example in each pair.

    Returns:
      A NumPy array of shape (2, 2) containing the pairwise accuracies, where the
      ij-th entry contains Acc_{G_i > G_j}.
    """
    raise NotImplementedError


def NDCG_predictor_model(estimator, X_test, y_test,
                         at=500, queries=False, k=1, m=1, use_ranker=False):
    """
    Calculates the normalized Discounted Cumulative Gain (nDCG) for a given ranking model (estimator).

    The function evaluates the ranking quality of the estimator on the provided test dataset. It can 
    operate in two modes: calculating nDCG for each query individually and averaging the results, or 
    calculating nDCG for the entire dataset as a whole. The nDCG calculation can be adjusted to 
    accommodate TREC-style evaluations.

    Parameters:
    - estimator: A machine learning model or estimator that provides a ranking or scoring of items.
    - x_test: The feature set of the test data, where each element corresponds to an item to be ranked.
    - y_test: The true relevance scores for the test data items. These are used as a reference for 
              calculating the ideal ranking.
    - at (int, optional): The number of top-ranked items to consider in the nDCG calculation. Defaults to 500.
    - queries (bool, optional): If True, the function calculates nDCG for each query in x_test separately 
                                and averages the scores. If False, it calculates nDCG for the entire test 
                                dataset. Defaults to False.
    - k (int, optional): A parameter used in the nDCG calculation, specifically in the nDCG_cls function. 
                         Defaults to 1.
    - m (int, optional): Another parameter for the nDCG calculation in the nDCG_cls function. Defaults to 1.
    - use_ranker (bool, optional): If True, a specific ranking-related method or attribute of the estimator 
                                   is used for predictions. If False, a standard prediction method is used. 
                                   Defaults to False.

    Returns:
    - float: The average nDCG score across all queries if queries is True, or the nDCG score for the entire 
             dataset if queries is False.

    Note:
    - The function is designed to be versatile, allowing for both query-wise and dataset-wide nDCG calculations.
    - The behavior of the nDCG calculation, especially in terms of handling TREC-style evaluations, is 
      determined by the nDCG_cls function.
    """
    if queries:
        # for trec at 30
        ndcg_list = []
        for x_q, y_q in zip(X_test, y_test):
            ndcg_list.append(nDCG_cls(estimator, x_q, y_q, at=at, trec=True, k=k, m=m))
        return np.mean(ndcg_list)
    else:
        if use_ranker:
            model_ndcg = nDCG_cls(estimator, X_test, y_test, at=at, trec=False, k=k, m=m)
        else:
            model_ndcg = nDCG_cls(estimator, X_test, y_test, at=at, trec=False, k=k, m=m)
        return model_ndcg


def NDCG_predictor_rf(estimator, X_test, y_test, y_bias_test, 
                      at=500, queries=False, k=1, m=1, user_ranker=False):
    """
    Calculates the ndcg for the repr. random forest
    """
    raise NotImplementedError


def NDCG_predictor_lr(estimator, X_test, y_test, y_bias_test,
                      at=500, queries=False, k=1, m=1, use_ranker=False):
    """
    Calculates the ndcg for the repr. linear model
    """
    raise NotImplementedError


def acc_fair_model(estimator, X_test, y_test, y_bias_test, 
                   queries=False, use_ranker=False, dataset=None):
    """
    Calculates the acc for the ranker output on the sensible attribute
    """
    raise NotImplementedError


def rND_estimator(estimator, X, s, step=10, start=10,
                  protected_group_idx=1, non_proteceted_group_idx=0,
                  y=None, esti=True):
    '''
    Computes the normalized Discounted Difference, which is a measure of how different are
    the proportion of members in the protected group at the top-i cutoff and in the overall
    population. Lower is better. 0 is the best possible value. Only binary protected groups
    are supported.
    '''
    raise NotImplementedError


def rND_torch(prediction, s, step=10, start=10, protected_group_idx=1, non_protected_group_idx=0):
    '''
    Computes the normalized Discounted Difference in PyTorch. This metric measures the disparity in 
    ranking outcomes between protected and non-protected groups in a binary classification context.
    Lower values indicate less disparity.

    Parameters:
    - prediction (torch.Tensor): The model predictions or scores.
    - s (torch.Tensor or list): The group labels (0 or 1), indicating whether each instance belongs 
                                to the protected group or not.
    - step (int): The step size to evaluate the ranking at different cutoffs.
    - start (int): The starting point for evaluating the ranking.
    - protected_group_idx (int): The index representing the protected group in `s`.
    - non_protected_group_idx (int): The index representing the non-protected group in `s`.

    Returns:
    - float: The normalized Discounted Difference score.
    '''

    # Ensure s is a 1D tensor
    #s = torch.as_tensor(s).flatten()

    # Check for size mismatch
    if len(prediction) != len(s):
        raise AssertionError(f'len of prediction {len(prediction)} and s {len(s)} are unequal')

    # Count occurrences of each group
    unique, counts = torch.unique(s, return_counts=True, sorted=True)
    count_dict_all = {k.item(): v.item() for k, v in zip(unique, counts)}

    # Ensure both groups are represented
    keys = [protected_group_idx, non_protected_group_idx]
    for key in keys:
        if key not in count_dict_all:
            count_dict_all[key] = 0

    # Sort predictions and corresponding group labels
    sorted_indices = torch.argsort(prediction, descending=True, dim=0)
    sorted_s = s[sorted_indices]

    # Create 'worst-case' sorted lists for regularization
    # first only the non protected group
    fake_horrible_s = torch.cat([torch.full((count_dict_all[non_protected_group_idx],), non_protected_group_idx),
                                 torch.full((count_dict_all[protected_group_idx],), protected_group_idx)])

    # first only the protected group
    fake_horrible_s_2 = torch.cat([torch.full((count_dict_all[protected_group_idx],), protected_group_idx),
                                   torch.full((count_dict_all[non_protected_group_idx],), non_protected_group_idx)])

    rnd, max_rnd, max_rnd_2 = 0.0, 0.0, 0.0

    for i in range(start, len(s), step):
        # Count occurrences in top i of the sorted list
        unique, counts = torch.unique(sorted_s[:i], return_counts=True)
        count_dict_top_i = {k.item(): v.item() for k, v in zip(unique, counts)}

        unique, counts = torch.unique(fake_horrible_s[:i], return_counts=True)
        count_dict_reg = {k.item(): v.item() for k, v in zip(unique, counts)}

        unique_2, counts_2 = torch.unique(fake_horrible_s_2[:i], return_counts=True)
        count_dict_reg_2 = {k.item(): v.item() for k, v in zip(unique_2, counts_2)}

        for key in keys:
            if key not in count_dict_reg:
                count_dict_reg[key] = 0
            if key not in count_dict_top_i:
                count_dict_top_i[key] = 0

        # Update rnd and max_rnd
        rnd += 1 / np.log2(i) * abs(
            count_dict_top_i[protected_group_idx] / i - count_dict_all[protected_group_idx] / len(s))
        max_rnd += 1 / np.log2(i) * abs(
            count_dict_reg[protected_group_idx] / i - count_dict_all[protected_group_idx] / len(s))
        max_rnd_2 += 1 / np.log2(i) * abs(
            count_dict_reg_2[protected_group_idx] / i - count_dict_all[protected_group_idx] / len(s))

    max_rnd = max(max_rnd, max_rnd_2)

    return rnd / max_rnd if max_rnd != 0 else 0


def transform_pairwise(X, y, s=None, subsample=0.):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    s_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        if s is not None:
            s_new.append((s[i, 0] - s[j, 0]) ** 2)
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    X_new = np.asarray(X_new)
    y_new = np.asarray(y_new).ravel()
    s_new = np.asarray(s_new)
    idx = [idx for idx in range(len(X_new))]
    if subsample > 0.0:
        idx = np.random.randint(X.shape[0], size=int(X.shape[0] * subsample))
        X_new = X_new[idx]
        y_new = y_new[idx]
        if s is not None:
            s_new = s_new[idx]

    if s is not None:
        return np.asarray(X_new), np.asarray(y_new).ravel(), idx, np.asarray(s_new)
    else:
        return np.asarray(X_new), np.asarray(y_new).ravel(), idx
