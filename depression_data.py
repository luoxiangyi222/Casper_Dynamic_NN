# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

import pandas as pd
import data_preprocessing as pre
import torch

# ##########################################################################################
# load raw data
# ##########################################################################################
gsr_data = pd.read_excel('depression/gsr_features.xlsx').iloc[:, 1:]
st_data = pd.read_excel('depression/skintemp_features.xlsx').iloc[:, 1:]
pd_data = pd.read_excel('depression/pupil_features.xlsx').iloc[:, 1:]
all_ft_data = pd.concat([gsr_data, st_data.iloc[:, 1:], pd_data.iloc[:, 1:]], axis=1)

gsr_data = pre.df_to_float_tensor(gsr_data)
st_data = pre.df_to_float_tensor(st_data)
pd_data = pre.df_to_float_tensor(pd_data)
all_ft_data = pre.df_to_float_tensor(all_ft_data)


# separate data into training and testing
# leave one participant out
NUM_PARTICIPANT = 12


def leave_one_participant_out(data: torch.Tensor, normalize_flag):
    """
    Apply leave-one-participant-out for input data, segmentation in this way
    For cross validation!
    return: normalised training data and testing data, in total 12 pairs
    """

    # 12 different participants hence 12 pairs of training and testing data
    train_data_list = []
    test_data_list = []

    for test_p in range(NUM_PARTICIPANT):
        # print('participant #' + str(test_p) + ' as testing person')
        start = 16 * test_p
        end = start + 16
        train_data = torch.cat([data[:start], data[end:]])
        test_data = data[start:end]

        # Normalization
        train_data = pre.normalization(train_data, normalize_flag)
        test_data = pre.normalization(test_data, normalize_flag)

        train_data_list.append(train_data)
        test_data_list.append(test_data)

    return train_data_list, test_data_list


