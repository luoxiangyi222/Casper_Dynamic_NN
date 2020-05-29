# comp4660 assignment 2 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020
from abc import abstractmethod
import numpy as np
import random


class GA(object):

    def __init__(self,
                 dna_size,
                 pop_size,
                 ):
        self.POP_SIZE = pop_size

        self.DNA_SIZE = dna_size

        self.died_pool = []  # record the used DNA, we do not want to reproduce them again

        self.old_candidate_pool = {}  # key: chromosome, value: select probability (initial zero)
        self.old_candidate_eval = {}

        self.new_candidate_pool = {}

        self.hall_of_fame = {}  # max size 20, worst will be killed
        self.fame_size = 20
        self.fame_eval = {}  # store evaluation measure matrix for best two
        self.fame_average_fit_list = []  # record average fit for each generation

        # model output
        self.best_fame = None
        self.best_fame_eval = None

        self.generation_counter = 0

        # rate
        self.CROSS_RATE = 0.8
        self.MUTATE_RATE = (1 / 240) + 0.11375   # decrease exponentially with generation counter

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

    def update_mutate_rate(self):
        self.MUTATE_RATE = (1 / 240) + 0.11375 / (2 ** self.generation_counter)


    def check_terminate(self):
        """
        Check last 3 iterations, if changes are all small, then terminate
        @return:
        """
        if len(self.fame_average_fit_list) < 4:
            return False
        else:
            last_five = np.array(self.fame_average_fit_list[-3:])
            shift_five = np.array(self.fame_average_fit_list[-4:-1])

            diff = abs(last_five - shift_five) < 0.005
            no_change = np.sum(diff)
            return no_change >= 3

    def reset_new_pool(self):
        self.new_candidate_pool = {}

    def rank_candidate(self):
        """
        @return: sorted candidate in ascending order
        """
        self.old_candidate_pool = {k: v for k, v in sorted(self.old_candidate_pool.items(), key=lambda item: item[1])}

    def rank_fame(self):
        """
        @return: sorted fame in ascending order
        """
        self.hall_of_fame = {k: v for k, v in sorted(self.hall_of_fame.items(), key=lambda item: item[1])}

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

    def record_fame_average_fitness(self):
        fitness_vals = list(self.hall_of_fame.values())
        average_fit = np.average(fitness_vals)
        self.fame_average_fit_list.append(average_fit)
        return average_fit

    def update_fame(self):
        """
        Insert best two individual into hall of fame
        @return:
        """
        # insert best 2 in the hall of fame
        best_two = list(self.old_candidate_pool.items())[-2:]
        self.hall_of_fame.update(best_two)

        # sort them and record average
        self.rank_fame()
        # check overflow
        if len(self.hall_of_fame) > self.fame_size:
            self.hall_of_fame = dict(list(self.hall_of_fame.items())[-self.fame_size:])
        # record average
        self.record_fame_average_fitness()

        # record evaluation measures of best two
        for _, item in enumerate(best_two):
            k, _ = item
            self.fame_eval[k] = self.old_candidate_eval[k]

        # clear old pool eval
        self.reset_old_eval()

        print('----- CURRENT FAME -----')
        print(self.hall_of_fame.values())

    def select_parent(self):
        """
        Select parent to generate offsprings
        @return: a pair of parent
        """
        # select one parent in fame
        father_str, _ = random.choice(list(self.hall_of_fame.items()))
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
        print('First generate')
        print('Counter')
        print(self.generation_counter)

        for i in range(self.POP_SIZE):  # random generate candidate
            chromosome = np.random.choice([0, 1], size=(self.DNA_SIZE,))
            self.old_candidate_pool[str(chromosome)] = 0

        self.fitness_function()
        self.rank_candidate()
        self.update_fame()

        print('Average fame fit')
        print(self.fame_average_fit_list[-1])

        self.generation_counter += 1
        self.update_mutate_rate()

        # record current pool to died pool
        self.update_died_pool()

    def update_died_pool(self):
        self.died_pool += list(self.old_candidate_pool.keys())

    def reset_old_eval(self):
        self.old_candidate_eval = {}

    def reproduce(self):
        """
        Create new generation of candidate pool
        @return:
        """

        while len(self.new_candidate_pool) < self.POP_SIZE:  # while not enough children
            fa, ma = self.select_parent()
            children = self.get_children(fa, ma)

            children_names = []

            # add children to new pool
            for _, c in enumerate(children):

                # check c did not in died pool and did not already added before
                child_str = str(c)
                if child_str not in self.died_pool:
                    if child_str not in children_names:
                        # add into new pool
                        self.new_candidate_pool[str(c)] = 0

    def evolve(self):
        finish = False
        while not finish:
            print('========================================================')
            print('Counter')

            print(self.fame_average_fit_list[-1])

            # create new population
            self.reproduce()

            # replace old pool
            self.old_candidate_pool = self.new_candidate_pool

            # compute fitness values
            self.fitness_function()

            self.rank_candidate()  # sort them
            self.update_fame()
            self.reset_new_pool()

            print(self.generation_counter)
            print('Average fame fit')

            # update mutate
            self.generation_counter += 1
            self.update_mutate_rate()

            self.update_died_pool()

            # check termination
            finish = self.check_terminate()

        print('Evolution finish')

        self.best_fame = list(self.hall_of_fame.items())[-1]
        self.best_fame_eval = self.fame_eval[self.best_fame[0]]

        print(self.fame_average_fit_list)
        print(self.best_fame)
        print(self.best_fame_eval)
        return self.best_fame, self.best_fame_eval

    # @abstractmethod
    def fitness_function(self):
        """
        An abstract function, the fitness for FFNN and Casper are different
        Comopute fitness for each individual in old_candidate_pool
        @return: update old_fitness_list
        """

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
        return np.array(index)


# ga = GA(5, 10)
