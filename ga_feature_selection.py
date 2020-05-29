# comp4660 assignment 2 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

from ga_abstract import GA
import my_nn_model as ffnn
import depression_data as dp_data
import my_casper_model as casper
import time


class GAFeatureSelectionFFNN(GA):
    def __init__(self,
                 dna_size,
                 pop_size,
                 num_hidden_units,
                 data):
        self.HIDDEN_NUM = num_hidden_units
        self.data = data

        GA.__init__(self, dna_size, pop_size)

    def fitness_function(self):

        for _, dna in enumerate(self.old_candidate_pool.keys()):
            dna_arr = self.str_to_array(dna)
            to_keep = self.dna_to_index(dna_arr)
            keep_data = self.data[:, to_keep]
            ffnn_compare = ffnn.FFNNModelComparison(keep_data,
                                                    use_lda=False,
                                                    learning_rate=0.01,
                                                    normalization_flag=2,
                                                    epochs=3000,
                                                    hidden_num=self.HIDDEN_NUM)

            overall_eval_measure, overall_accuracy = ffnn_compare.final_evaluation()

            self.old_candidate_pool[dna] = overall_accuracy
            self.old_candidate_eval[dna] = overall_eval_measure



class GAFeatureSelectionCasper(GA):
    def __init__(self,
                 pop_size,
                 dna_size,
                 num_hidden_units,
                 data):

        self.HIDDEN_NUM = num_hidden_units
        self.data = data

        super(GAFeatureSelectionCasper, self).__init__(pop_size, dna_size)

    def fitness_function(self):
        """
        Evaluate the subset of features, record measures and accuracy
        @return:
        """

        for _, dna in enumerate(self.old_candidate_pool.keys()):
            dna_arr = self.str_to_array(dna)
            to_keep = self.dna_to_index(dna_arr)
            keep_data = self.data[:, to_keep]
            casper_compare = casper.CasPerModelComparison(data=keep_data,
                                                          use_lda=False,
                                                          normalization_flag=2,
                                                          hidden_num=self.HIDDEN_NUM)

            overall_eval_measure, overall_accuracy = casper_compare.final_evaluation()

            self.old_candidate_pool[dna] = overall_accuracy
            self.old_candidate_eval[dna] = overall_eval_measure


fff = GAFeatureSelectionFFNN(85, 20, 1, dp_data.all_ft_data)

# ccc = GAFeatureSelectionCasper(85, 20, 1, dp_data.all_ft_data)

###############3
# FFNN

# ffnn_GA_file = open("ffnn_GA.txt", "w+")
# ffnn_GA_file.write('All features \n')
# ffnn_GA_file.write('GA: True  \n')
# time_cost = []
# for hn in range(1, 20):
#     ffnn_GA_file.write('hidden units: ' + str(hn) + '\n')
#
#     start = time.time()
#     GA_ffnn = GAFeatureSelectionFFNN(85, 20, hn, dp_data.all_ft_data)
#
#     end = time.time()
#     time_cost.append(end-start)
#
#     best_fame = GA_ffnn.best_fame
#     best_fame_eval = GA_ffnn.best_fame_eval
#     ffnn_GA_file.write('best fame\n' + str(best_fame) + '\n')
#     ffnn_GA_file.write('best fame evaluation\n' + str(best_fame_eval) + '\n')
#
# ffnn_GA_file.write('time cost : \n' + str(time_cost))
# ffnn_GA_file.close()

##########################
# Casper

# casper_GA_file = open("ffnn_GA.txt", "w+")
# casper_GA_file.write('All features \n')
# casper_GA_file.write('GA: True  \n')
# time_cost = []
# for hn in range(1, 20):
#     casper_GA_file.write('hidden units: ' + str(hn) + '\n')
#
#     start = time.time()
#     GA_casper = GAFeatureSelectionCasper(85, 20, hn, dp_data.all_ft_data)
#
#     end = time.time()
#     time_cost.append(end-start)
#
#     best_fame = GA_casper.best_fame
#     best_fame_eval = GA_casper.best_fame_eval
#     casper_GA_file.write('best fame\n' + str(best_fame) + '\n')
#     casper_GA_file.write('best fame evaluation\n' + str(best_fame_eval) + '\n')
#
# casper_GA_file.write('time cost : \n' + str(time_cost))
# casper_GA_file.close()

