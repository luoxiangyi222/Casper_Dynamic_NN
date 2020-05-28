# comp4660 assignment 1 code
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

import evaluation
import data_preprocessing
import my_casper_model as casper
import my_nn_model as ffnn
import depression_data as dp_data
import time

# ##########################################################################################
# Feed Forward Network
# ##########################################################################################

"""
To run model just uncomment related section!!!!!!!!!!!!!
"""
# 1 all features

# ffnn_allfeature_result_file = open("ffnn_allfeature_result.txt", "w+")
# ffnn_allfeature_result_file.write('All features \n')
# ffnn_allfeature_result_file.write('LDA: False  \n')
#
# all_accuracy_list = []
# time_cost_list = []
# for hn in range(1, 20):
#     num_hidden = hn * 5
#     print('Hidden units: ' + str(num_hidden))
#     ffnn_allfeature_result_file.write('hidden units: ' + str(num_hidden) + '\n')
#     t1 = time.time()
#     ffnn_compare = ffnn.FFNNModelComparison(dp_data.all_ft_data, use_lda=False, learning_rate=0.001, normalization_flag=2,
#                                             epochs=3000, hidden_num=num_hidden)
#
#     t2 = time.time()
#     time_cost = t2 - t1
#     time_cost_list.append(time_cost)
#
#     eval_measures, accuracy = ffnn_compare.final_evaluation()
#     all_accuracy_list.append(accuracy)
#
#     ffnn_allfeature_result_file.write(str(eval_measures))
#     ffnn_allfeature_result_file.write('\n')
#     ffnn_allfeature_result_file.write(str(accuracy) + '\n')
#
# ffnn_allfeature_result_file.write('accuracy list: \n' + str(all_accuracy_list) + '\n')
# ffnn_allfeature_result_file.write('time cost list: \n' + str(time_cost_list) + '\n')
# print(all_accuracy_list)
#
# ffnn_allfeature_result_file.close()

# 2 all feature + LDA

ffnn_allfeature_LDA_result_file = open("ffnn_LDA_allfeature_result.txt", "w+")
ffnn_allfeature_LDA_result_file.write('All features \n')
ffnn_allfeature_LDA_result_file.write('LDA: True  \n')

all_accuracy_list = []
time_cost_list = []
for hn in range(1, 20):
    num_hidden = hn * 5
    print('Hidden units: ' + str(num_hidden))
    ffnn_allfeature_LDA_result_file.write('hidden units: ' + str(num_hidden) + '\n')
    t1 = time.time()
    ffnn_compare = ffnn.FFNNModelComparison(dp_data.all_ft_data, use_lda=True, learning_rate=0.01, normalization_flag=2,
                                            epochs=3000, hidden_num=num_hidden)
    # ffnn_compare.visualization()
    t2 = time.time()
    time_cost = t2 - t1
    time_cost_list.append(time_cost)

    eval_measures, accuracy = ffnn_compare.final_evaluation()
    all_accuracy_list.append(accuracy)

    ffnn_allfeature_LDA_result_file.write(str(eval_measures))
    ffnn_allfeature_LDA_result_file.write('\n')
    ffnn_allfeature_LDA_result_file.write(str(accuracy) + '\n')

ffnn_allfeature_LDA_result_file.write('accuracy list: \n' + str(all_accuracy_list) + '\n')
ffnn_allfeature_LDA_result_file.write('time cost list: \n' + str(time_cost_list) + '\n')
print(all_accuracy_list)

ffnn_allfeature_LDA_result_file.close()

# #######################################################################
# CasPer
# #######################################################################


# 1 all features
# casper_allfeature_result_file = open("casper_allfeature_result.txt", "w+")
# casper_allfeature_result_file.write('All features \n')
# casper_allfeature_result_file.write('LDA: False  \n')
#
# all_accuracy_list = []
# time_cost_list = []
# for hn in range(1, 20):
#     print('Hidden units: ' + str(hn))
#     casper_allfeature_result_file.write('hidden units: ' + str(hn) + '\n')
#     t1 = time.time()
#     casperCompare = casper.CasPerModelComparison(data=dp_data.all_ft_data, use_lda=False,
#                                                  normalization_flag=2, hidden_num=hn)
#     eval_measures, accuracy = casperCompare.final_evaluation()
#     all_accuracy_list.append(accuracy)
#     t2 = time.time()
#     time_cost = t2 - t1
#     time_cost_list.append(time_cost)
#
#     casper_allfeature_result_file.write(str(eval_measures))
#     casper_allfeature_result_file.write('\n')
#     casper_allfeature_result_file.write(str(accuracy) + '\n')
#
# casper_allfeature_result_file.write('accuracy list: \n' + str(all_accuracy_list) + '\n')
# casper_allfeature_result_file.write('time cost list: \n' + str(time_cost_list) + '\n')
# print(all_accuracy_list)
#
# casper_allfeature_result_file.close()


# 2 all feature + LDA
# compare number of hidden neurons

# casper_allfeature_lda_result_file = open("casper_allfeature_LDA_result.txt", "w+")
# casper_allfeature_lda_result_file.write('All features \n')
# casper_allfeature_lda_result_file.write('LDA: true  \n')
#
# all_lda_accuracy_list = []
# time_cost_list = []
# for hn in range(1, 20):
#     print('Hidden units: ' + str(hn))
#     casper_allfeature_lda_result_file.write('hidden units: ' + str(hn) + '\n')
#     t1 = time.time()
#     casperCompare = casper.CasPerModelComparison(data=dp_data.all_ft_data, use_lda=True,
#                                                  normalization_flag=2, hidden_num=hn)
#     eval_measures, accuracy = casperCompare.final_evaluation()
#     all_lda_accuracy_list.append(accuracy)
#     t2 = time.time()
#     time_cost = t2 - t1
#     time_cost_list.append(time_cost)
#
#     casper_allfeature_lda_result_file.write(str(eval_measures))
#     casper_allfeature_lda_result_file.write('\n')
#     casper_allfeature_lda_result_file.write(str(accuracy) + '\n')
#
# casper_allfeature_lda_result_file.write('accuracy list: \n' + str(all_lda_accuracy_list) + '\n')
# casper_allfeature_lda_result_file.write('time cost list: \n' + str(time_cost_list) + '\n')
# print(all_lda_accuracy_list)
#
# casper_allfeature_lda_result_file.close()
