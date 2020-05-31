# comp4660 assignment 2 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

"""
There will be 3 different data input type, including:
1 all features
2 all features + LDA
3 all features + GA


Each of these data will be put into:
1. the feed forward network implemented by myself
2. Casper model described in technique paper by Prof. Golden Tom

The goal of this module is to compare performance of models given different preprocessed data
"""


import my_casper_model as casper
import my_nn_model as ffnn
import depression_data as dp_data
import ga_feature_selection as gafs

# ##########################################################################################
# Feed Forward Network
# ##########################################################################################

"""
To run model just uncomment related section!!!!!!!!!!!!!
"""
# 1 FFNN

# ffnn_compare = ffnn.FFNNModelComparison(dp_data.all_ft_data, use_lda=False, learning_rate=0.01, normalization_flag=2,
#                                         epochs=3000, hidden_num=5)
# ffnn_compare.visualization()
# ffnn_compare.final_evaluation()


# 2 FFNN + LDA

# ffnn_compare = ffnn.FFNNModelComparison(dp_data.all_ft_data, use_lda=True, learning_rate=0.01, normalization_flag=2,
#                                         epochs=3000, hidden_num=5)
# ffnn_compare.visualization()
# ffnn_compare.final_evaluation()


# 3 FFNN + GA

# ffnn_ga = gafs.GAFeatureSelectionFFNN(dna_size=85, pop_size=10, num_hidden_units=5, data=dp_data.all_ft_data)

# #######################################################################
# CasPer
# #######################################################################


# 4 Casper


# casperCompare = casper.CasPerModelComparison(data=dp_data.all_ft_data, use_lda=False,
#                                              normalization_flag=2, hidden_num=5, display=True)
# eval_measures, accuracy = casperCompare.final_evaluation()


# 5 all feature + LDA

casperCompare = casper.CasPerModelComparison(data=dp_data.all_ft_data, use_lda=True,
                                             normalization_flag=2, hidden_num=5, display=True)
eval_measures, accuracy = casperCompare.final_evaluation()


# 3 Casper + GA

# casper_ga = gafs.GAFeatureSelectionCasper(dna_size=85, pop_size=10, num_hidden_units=5, data=dp_data.all_ft_data)
