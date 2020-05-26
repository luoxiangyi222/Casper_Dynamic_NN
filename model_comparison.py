# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

"""
There will be 8 different data input type, including:
1 all features + normalization
2 GSR + normalization
3 ST + normalization
4 PD + normalization

5 all features + LDA + normalization
6 GSR + LDA + normalization
7 ST + LDA + normalization
8 PD + LDA + normalization


Each of these data will be put into:
1. the feed forward network implemented by myself
2. Cascade NN (modified version of Casper)

The goal of this is to compare performance of models given different preprocessed data
"""

import evaluation
import data_preprocessing
import my_casper_model as casper
import my_nn_model as ffnn
import depression_data as depr_data




# ##########################################################################################
# Feed Forward Network
# ##########################################################################################

"""
To run model just uncomment related section!!!!!!!!!!!!!
"""

# 1 all features + normalization + FFNN
# all_ft_model = ffnn.FFNNModelComparison(data.all_ft_data, use_lda=False,
#                                    learning_rate=0.05,
#                                    normalization_flag=0,
#                                    epochs=300,
#                                    hidden_num=50)
# all_ft_model.visualization()
# all_ft_model.final_evaluation()

# 2 GSR + normalization + FFNN
# gsr_model = ffnn.FFNNModelComparison(gsr_data,
#                                      use_lda=False,
#                                      learning_rate=0.1,
#                                      normalization_flag=2,
#                                      epochs=3000,
#                                      hidden_num=10)
# gsr_model.visualization()
# gsr_model.evaluation()

# 3 ST + normalization + FFNN
# st_model = ffnn.FFNNModelComparison(data.st_data,
#                                     use_lda=False,
#                                     learning_rate=0.01,
#                                     normalization_flag=0,
#                                     epochs=3000,
#                                     hidden_num=10)
# st_model.visualization()
# st_model.final_evaluation()

# 4 PD + normalization + FFNN
# pd_model = ffnn.FFNNModelComparison(pd_data,
#                                     use_lda=False,
#                                     learning_rate=0.1,
#                                     normalization_flag=2,
#                                     epochs=3000,
#                                     hidden_num=10)
# pd_model.visualization()
# pd_model.evaluation()

# 5 all features + LDA + normalization
# all_ft_lda_model = ffnn.FFNNModelComparison(all_ft_data,
#                                             use_lda=True,
#                                             learning_rate=0.1,
#                                             normalization_flag=2,
#                                             epochs=3000,
#                                             hidden_num=10)
# all_ft_lda_model.visualization()
# all_ft_lda_model.evaluation()

# 6 GSR + LDA + normalization
# gsr_lda_model = ffnn.FFNNModelComparison(gsr_data,
#                                          use_lda=True,
#                                          learning_rate=0.1,
#                                          normalization_flag=2,
#                                          epochs=3000,
#                                          hidden_num=10)
# gsr_lda_model.visualization()
# gsr_lda_model.evaluation()

# 7 ST + LDA + normalization
#
# st_lda_model = ffnn.FFNNModelComparison(st_data,
#                                         use_lda=True,
#                                         learning_rate=0.1,
#                                         normalization_flag=2,
#                                         epochs=3000,
#                                         hidden_num=10)
# st_lda_model.visualization()
# st_lda_model.evaluation()

# 8 PD + LDA + normalization

# pd_lda_model = ffnn.FFNNModelComparison(pd_data,
#                                         use_lda=True,
#                                         learning_rate=0.1,
#                                         normalization_flag=2,
#                                         epochs=3000,
#                                         hidden_num=10)
# pd_lda_model.visualization()
# pd_lda_model.evaluation()

# all_ft_casper = casper.CasPerModelComparison(all_ft_data,
#                                              use_lda=False,
#                                              normalization_flag=2)
# all_ft_casper.visualization()
# all_ft_casper.evaluation()


# #######################################################################
# CasPer
# #######################################################################


# 1 all features + normalization
# all_ft_casper = casper.CasPerModelComparison(all_ft_data,
#                                              use_lda=False,
#                                              normalization_flag=2)
# all_ft_casper.visualization()
# all_ft_casper.evaluation()

# 2 GSR + normalization
# gsr_casper = casper.CasPerModelComparison(gsr_data,
#                                           use_lda=False,
#                                           normalization_flag=2)
# gsr_casper.visualization()
# gsr_casper.evaluation()
#
# # 3 ST + normalization
#
# st_casper = casper.CasPerModelComparison(st_data,
#                                          use_lda=False,
#                                          normalization_flag=2)
# st_casper.visualization()
# st_casper.evaluation()
#
# # 4 PD + normalization
#
#
# pd_casper = casper.CasPerModelComparison(pd_data,
#                                          use_lda=False,
#                                          normalization_flag=2)
# pd_casper.visualization()
# pd_casper.evaluation()
#
# #
# # 5 all features + LDA + normalization
#
# all_ft_LDA_casper = casper.CasPerModelComparison(all_ft_data,
#                                                  use_lda=True,
#                                                  normalization_flag=2)
# all_ft_LDA_casper.visualization()
# all_ft_LDA_casper.evaluation()
#
# # 6 GSR + LDA + normalization
# gsr_LDA_casper = casper.CasPerModelComparison(gsr_data,
#                                               use_lda=True,
#                                               normalization_flag=2)
# gsr_LDA_casper.visualization()
# gsr_LDA_casper.evaluation()
#
# # 7 ST + LDA + normalization
#
# st_LDA_casper = casper.CasPerModelComparison(data.st_data,
#                                              use_lda=True,
#                                              normalization_flag=2)
# st_LDA_casper.visualization()
# st_LDA_casper.evaluation()
#
# # 8 PD + LDA + normalization
# pd_LDA_casper = casper.CasPerModelComparison(depr_data.pd_data,
#                                              use_lda=True,
#                                              normalization_flag=2)
# pd_LDA_casper.visualization()
# pd_LDA_casper.evaluation()