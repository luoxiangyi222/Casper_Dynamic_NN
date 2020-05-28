# comp4660 assignment 2 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

from GA_Abstract import GA
import numpy as np
import my_nn_model as ffnn
import depression_data as dp_data


all_df_data = dp_data.all_ft_data


class GAFeatureSelectionCasper(GA):
    def __init__(self,
                 pop_size,
                 dna_size):
        super(GAFeatureSelectionCasper, self).__init__(pop_size, dna_size)


    def fitness_function(self):
        for k, dna in enumerate(self.old_candidate_pool.keys()):
            dna_arr = self.str_to_array(dna)
            print(dna_arr)
            to_keep = self.dna_to_index(dna_arr)
            print(to_keep)
            keep_data = all_df_data[:, to_keep]

            ffnn_compare = ffnn.FFNNModelComparison(keep_data, use_lda=False, learning_rate=0.01,
                                                    normalization_flag=2,
                                                    epochs=3000, hidden_num=20)

            _, overall_accuracy = ffnn_compare.final_evaluation()

            self.old_candidate_pool[dna] = overall_accuracy


class GAFeatureSelectionFFNN(GA):
    def __init__(self,
                 dna_size,
                 pop_size):

        super().__init__(dna_size, pop_size)

    @staticmethod
    def dna_to_index(array):
        """
        Change dna to index of kept features
        @param array:
        @return: index of feature keep
        """
        index = []
        for i in range(len(array)):
            if array[i] == 1:
                index.append(i)

        index = [0] + [x+1 for x in index]
        return index

    def fitness_function(self):
        for k, dna in enumerate(self.old_candidate_pool.keys()):
            dna_arr = self.str_to_array(dna)
            print(dna_arr)
            to_keep = self.dna_to_index(dna_arr)
            print(to_keep)
            keep_data = all_df_data[:, to_keep]

            ffnn_compare = ffnn.FFNNModelComparison(keep_data, use_lda=False, learning_rate=0.01,
                                                    normalization_flag=2,
                                                    epochs=3000, hidden_num=20)

            _, overall_accuracy = ffnn_compare.final_evaluation()

            self.old_candidate_pool[dna] = overall_accuracy


# GA_ffnn = GAFeatureSelectionFFNN(85, 10)

GA_casper = GAFeatureSelectionCasper