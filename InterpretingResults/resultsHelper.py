import pandas as pd
import numpy as np
from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.stats.power as smp


def perform_shapiro_test(df, column):
    """
    Performs The Shapiro Wilk test to check if the data is normal

    Arguments:
    df (pandas.Dataframe): 
    column (str): The name of the column

    Returns:
    None
    """
    schedules = df['Schedule'].unique()
    for schedule in schedules:
        stat, p_value = shapiro(df[df['Schedule'] == schedule][column])
        print(f"Shapiro-Wilk Test for {column} in Schedule {schedule}: Statistic={stat}, p-value={p_value}")
        if p_value < 0.05:
            print("\tNull hypothesis rejected - data is not normal")
        else:
            print("\tNull hypothesis not rejected - data is normal")


def cohen_d(group1, group2):
    """
    A function which calculates Cohen's d which is a measure for the
    effect size

    Parameters:
    group1: List or Numpy Array
    group2: List or Numpy Array

    Returns:
    d (float): Cohen's d 
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    n1, n2 = len(group1), len(group2)
    
    # Calculate the pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d


def pearson_cc(group1, group2):
    # TODO: wrong
    """
    Calculates the Pearson Correlation Coefficient

    Parameters:
    group1: List or Numpy Array
    group2: List or Numpy Array

    Returns:
    r (float): Pearson Correlation Coefficient
    """
    # Calculate Cohen's d
    d = cohen_d(group1, group2)
    # Calculate Pearson Correlation Coefficient
    r = d / np.sqrt(d**2+4)
    return r
