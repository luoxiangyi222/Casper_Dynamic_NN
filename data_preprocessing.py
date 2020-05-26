# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

"""
Data pre-processing is very important.
This module provide some pre processing functions
As we have small data set for depression, the LDA was used for dimension reduction.
In this assignment, I want to show that how dimension reduction technique will affect the model performance.

Also, we should do pre-processing for training data and testing data separately!!!!!!!

"""

import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ##########################################################################################
# some basic functions
# ##########################################################################################


def normalization(data_tensor, normal_function_flag=0):
    """
    :param data_tensor: data , the first column is label
    :param normal_function_flag: 0 for minmax, 1 for maxabs, 2 for robust, default is 0
    :return: normalized dataFrame, the first column unchanged
    """

    X = data_tensor[:, 1:]  # features
    y = data_tensor[:, 0].unsqueeze(dim=1)  # label

    scaler = preprocessing.MinMaxScaler()
    if normal_function_flag == 1:
        scaler = preprocessing.MaxAbsScaler()
    if normal_function_flag == 2:
        scaler = preprocessing.RobustScaler()

    normal_X = torch.from_numpy(scaler.fit_transform(X)).float()

    normal_tensor = torch.cat([y, normal_X], dim=1)
    return normal_tensor


def lda_feature_selection(data_tensor, num_features):
    """
    :param data_tensor: data , the first column is label
    :param num_features: number of features after lda, should be less than number of class
    :return: lda features, the first column unchanged
    """
    # Use LDA to reduce the data down to 3 features
    lda = LinearDiscriminantAnalysis(n_components=num_features)
    X = data_tensor[:, 1:]
    y = data_tensor[:, 0]
    lda_X = torch.from_numpy(lda.fit(X, y).transform(X)).float()
    y = y.unsqueeze(dim=1)
    lda_data_df = torch.cat([y, lda_X], dim=1)
    return lda_data_df


def remove_colinear_features(data_tensor, drop_threshold):
    """
    :param data_tensor: dataFrame, the first column is label
    :param drop_threshold: features correlation greater than threshold will be removed
    :return: removing high correlated features, the first column unchanged
    """

    X = data_tensor[:, 1:]
    y = data_tensor[:, 0].unsqueeze(dim=1)
    X_df = pd.DataFrame(X.numpy())
    matrix_abs_corr = X_df.corr().abs()

    upper = matrix_abs_corr.where(np.triu(np.ones(matrix_abs_corr.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > drop_threshold)]
    dropped_X_df = X_df.drop(X_df[to_drop], axis=1)
    dropped_X = torch.from_numpy(dropped_X_df.values).float()
    dropped_X = torch.cat([y, dropped_X], dim=1)
    return to_drop, dropped_X


def add_bias_layer(data_tensor):
    """
    :param data: the first column is label
    :return: add last column all ones, the first column unchanged
    """
    rows, cols = data_tensor.shape
    bias_column = torch.ones((rows, 1))
    data_bias_tensor = torch.cat([data_tensor, bias_column], dim=1)

    return data_bias_tensor


def df_to_float_tensor(df):
    tensor = torch.from_numpy(df.values).float()
    return tensor


def arr_to_float_tensor(arr):
    tensor = torch.from_numpy(arr).float()
    return tensor





