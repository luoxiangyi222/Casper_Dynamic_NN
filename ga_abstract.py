# comp4660 assignment 2 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020
from abc import abstractmethod
import numpy as np
import random


class GA(object):

    def __init__(self,
                 dna_size,
                 pop_size=10,
                 ):
        self.POP_SIZE = pop_size

        self.DNA_SIZE = dna_size

        self.old_candidate_pool = {}  # key: chromosome, value: select probability (initial zero)

        self.new_candidate_pool = {}

        self.frame_size = 4
        self.hall_of_frame = {}  # max size 20, worst will be killed

        self.CROSS_RATE = 0.8
        self.MUTATE_RATE = 0.002

        self.average_fit_list = []

        self.chosen_solution = None

        self.generation_counter = 0
        self.first_generate()
        self.evolve()

    @staticmethod
    def str_to_array(str_of_array):
        """
        In pool dict, key is a string, need to convert to np array
        @param str_of_array:
        @return:
        """
        str_of_array = str_of_array.strip('][').split(' ')
        int_array = np.array(str_of_array).astype(np.int)
        return int_array

    def get_average_fitness(self):
        fitness_vals = list(self.old_candidate_pool.values())
        average_fit = np.average(fitness_vals)
        return average_fit

    def check_terminate(self):
        """
        Check last 5 iterations, if changes are all small, then terminate
        @return:
        """
        if len(self.average_fit_list) < 6:
            return False
        else:
            last_five = np.array(self.average_fit_list[-5:])
            shift_five = np.array(self.average_fit_list[-6:-1])

            diff = abs(last_five - shift_five) < 0.1
            no_change = np.sum(diff)
            return no_change >= 5


    def reset_new_pool(self):
        self.new_candidate_pool = {}

    def rank_candidate(self):
        """
        @return: sorted candidate in ascending order
        """
        self.old_candidate_pool = {k: v for k, v in sorted(self.old_candidate_pool.items(), key=lambda item: item[1])}

    def rank_frame(self):
        """
        @return: sorted frame in ascending order
        """
        self.hall_of_frame = {k: v for k, v in sorted(self.hall_of_frame.items(), key=lambda item: item[1])}

    def rank_based_select_from_old_pool(self):
        """
        reference: http://www.geatbx.com/docu/algindex-02.html#P244_16021
        Select a parent based on rank
        @return: array, one female parent
        """
        # least fit individual has Pos=1, the fittest individual Pos=N

        SP = 1.5  # select pressure
        N = len(self.old_candidate_pool)
        pos_arr = np.array(list(range(N)))
        keys_list = list(self.old_candidate_pool.keys())

        probability = (2-SP) + 2*(SP-1)*(pos_arr - 1) / (N-1)

        female_parent = random.choices(keys_list, probability)[0]
        female_parent = self.str_to_array(female_parent)

        return female_parent

    def update_frame(self):
        """
        Insert best two individual into hall of frame
        @param best_two:
        @return:
        """
        # insert best 2 in the hall of fame
        best_two = list(self.old_candidate_pool.items())[-2:]

        self.hall_of_frame.update(best_two)
        self.rank_frame()

        # check overflow
        if len(self.hall_of_frame) > self.frame_size:
            self.hall_of_frame = dict(list(self.hall_of_frame.items())[-self.frame_size:])

        print('-----FRAME---------')
        print(self.hall_of_frame)

    def select_parent(self):
        """
        Select parent to generate offsprings
        @return: a pair of parent
        """
        # select one parent in frame
        father_str, _ = random.choice(list(self.hall_of_frame.items()))
        father_arr = self.str_to_array(father_str)

        # select one parent in current pool based on rank
        mother_arr = self.rank_based_select_from_old_pool()

        return father_arr, mother_arr

    def crossover(self, father, mother):
        """
        @param father: array
        @param mother: array
        @return: two children in list
        """
        if np.random.rand() < self.CROSS_RATE:
            mask = np.random.choice([0, 1], size=(self.DNA_SIZE,))

            fa_and_mask = father & mask
            fa_and_not_mask = father & ~mask

            ma_and_mask = mother & mask
            ma_and_not_mask = mother & ~mask

            offspring1 = fa_and_mask | ma_and_not_mask
            offspring2 = fa_and_not_mask | ma_and_mask
            children = [offspring1, offspring2]

        else:
            children = [father, mother]

        return children

    def mutate(self, child: np.array):
        """
        mutation
        @return: mutated child
        """
        for point in range(self.DNA_SIZE):
            if np.random.rand() < self.MUTATE_RATE:
                child[point] = 1 if child[point] == 0 else 0
        return child

    def get_children(self, father, mother):
        """
        helper function for reproduce
        @param father: array
        @param mother: array
        @return: two children
        """
        children = self.crossover(father, mother)
        mutate_children = []
        for i, c in enumerate(children):
            mutate_c = self.mutate(c)
            mutate_children.append(mutate_c)
        return mutate_children

    def first_generate(self):
        """
        @return:
        """
        for i in range(self.POP_SIZE):  # random generate candidate
            chromosome = np.random.choice([0, 1], size=(self.DNA_SIZE,))
            self.old_candidate_pool[str(chromosome)] = 0

        self.fitness_function()
        self.rank_candidate()
        self.update_frame()

        average_fit = self.get_average_fitness()
        self.average_fit_list.append(average_fit)

        self.generation_counter += 1

    def reproduce(self):
        """
        Create new generation of candidate pool
        @return:
        """

        while len(self.new_candidate_pool) < self.POP_SIZE:  # while not enough children
            fa, ma = self.select_parent()
            children = self.get_children(fa, ma)

            # add children to new pool
            for _, c in enumerate(children):

                # check new pool do not have repeat children
                new_keys_list = list(self.new_candidate_pool.keys())
                if len(new_keys_list) > 0:

                    if str(c) in new_keys_list:
                        break

                self.new_candidate_pool[str(c)] = 0

    def evolve(self):
        finish = False
        while not finish:
            print('========================================================')
            print('Counter')
            print(self.generation_counter)
            print('Average fit')
            print(self.average_fit_list[-1])

            # create new population
            self.reproduce()

            # replace old pool
            self.old_candidate_pool = self.new_candidate_pool

            # compute fitness values
            self.fitness_function()
            self.rank_candidate()
            self.update_frame()
            self.reset_new_pool()

            # record average fit
            average_fit = self.get_average_fitness()
            self.average_fit_list.append(average_fit)

            self.generation_counter += 1

            # check termination
            finish = self.check_terminate()

        print('Evolution finish')
        print(self.average_fit_list)
        self.chosen_solution = list(self.old_candidate_pool.items())[-2:]
        print(self.chosen_solution)
        return self.chosen_solution

    # @abstractmethod
    def fitness_function(self):
        """
        An abstract function, the fitness for FFNN and Casper are different
        Comopute fitness for each individual in old_candidate_pool
        @return: update old_fitness_list
        """
        # model_compare = ffnn.FFNNModelComparison(self.data, True, 0.001, 2, 3000, 5)
        # for i, dna in enumerate(self.old_candidate_pool.keys()):
        #     dna_arr = self.str_to_array(dna)
        #     fitness = np.sum(dna_arr)
        #     self.old_candidate_pool[dna] = fitness
        pass

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

