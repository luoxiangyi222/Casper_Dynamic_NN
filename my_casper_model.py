# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

"""
This module implemented the Casper model
"""


import torch
import torch.nn as nn
import data_preprocessing as data_pre
import matplotlib.pyplot as plt
import evaluation as eval
import depression_data as dp_data
import pdb


# ##############################################################################################
# Build CasPer model
# ##############################################################################################
dtype = torch.float
device = torch.device("cpu")


class SARPROP(object):
    """
    THis class is a optimizer for SA RPROP
    """

    def __init__(self,
                 params: list,
                 region_flag: int,  # determine initial step
                 etas=(0.5, 1.2),
                 step_sizes=(1e-6, 50)
                 ):

        self.params = params
        self.steps = [0.2, 0.005, 0.001]
        self.initial_step = self.steps[region_flag]
        self.etas = etas
        self.step_sizes = step_sizes

        # hyper parameter
        self.k1 = 1e-4
        self.T = 0.01
        self.state = {}  # store state, for instance, previous gradient, step_size
        for i in range(len(params)):
            self.state[i] = {}
        self.step_counter = 0

    def zero_grad(self):
        """
        set all grad to zero
        @return:
        """
        if self.params:
            for par_id, p in enumerate(self.params):
                p.grad.zero_()

    def step(self):
        """
        Update state and weight

        """
        self.step_counter += 1

        for par_id, p in enumerate(self.params):

            grad = p.grad
            state = self.state[par_id]

            # initialization of the state
            if len(state) == 0:
                state['prev_grad'] = torch.zeros_like(p)
                state['step_size'] = torch.ones_like(grad) * self.initial_step

            etaminus, etaplus = self.etas
            step_size_min, step_size_max = self.step_sizes
            step_size = state['step_size']

            # compute SA gradient
            sa_term = self.k1 * (2 ** (- self.T * self.step_counter)) * (p.sign()) * p * p

            SA_grad = grad - sa_term
            check_same_sign = (grad * SA_grad).sign()

            sign = (SA_grad * (state['prev_grad'])).sign()

            step = torch.ones_like(sign)

            # set factor according to sign
            step[sign.gt(0)] = etaplus
            step[sign.lt(0)] = etaminus
            step[sign.eq(0)] = 1

            # update stepsizes with step size updates
            step_size.mul_(step).clamp_(step_size_min, step_size_max)

            # for dir<0, dfdx=0
            # for dir>=0 dfdx=dfdx
            SA_grad = SA_grad.clone(memory_format=torch.preserve_format)
            # SA_grad[check_same_sign.lt(0)] = step_size_min
            SA_grad[step.eq(etaminus)] = 0

            # update parameters
            with torch.no_grad():
                p.addcmul_(SA_grad.sign(), step_size, value=-1)

            # save previous gradient
            state['prev_grad'] = SA_grad.clone().detach()


class HiddenNeuron(object):
    """
    This class represents a local structure in Casper network,
    A hidden Neuron includes weight in, sum of all input, and one output
    """

    def __init__(self,
                 num_in: int,
                 ):
        # randomly initialise the in and out weights
        self.w_in = torch.randn(num_in, 1, device=device, dtype=dtype, requires_grad=True)

        self.y_in = None
        self.y_out = None

    def compute_y_in(self, input_layer):
        self.y_in = torch.mm(input_layer, self.w_in)

    def compute_y_out(self, input_layer):
        self.compute_y_in(input_layer)
        self.y_out = torch.tanh(self.y_in)  # activation function
        return self.y_out


class CasPerModel(object):

    def __init__(self,
                 input_d: int,
                 output_d: int,
                 train_data: torch.Tensor,
                 test_data: torch.Tensor,
                 num_hidden,  # hyper parameter, how many hidden neurons will be added in
                 model_id
                 ):
        self.id = model_id
        # input, output dimension
        self.input_d = input_d
        self.output_d = output_d
        self.output_layer = None

        # hyper parameter
        self.NUM_HIDDEN_NEURON = num_hidden
        self.P = 0.5

        # store all data and test data, first column is label
        self.train_x = train_data[:, 1:]
        self.train_y = train_data[:, 0].long()

        self.test_x = test_data[:, 1:]  # one participant out
        self.test_y = test_data[:, 0].long()

        self.all_train_loss = []  # each element contains all training loss in one training for one hidden neuron
        self.all_test_loss = []  # each element contains all testing loss in one training  for one hidden neuron
        self.all_train_accuracy = []  # each element contains all training loss in one training for one hidden neuron
        self.all_test_accuracy = []  # each element contains all testing loss in one training  for one hidden neuron

        # neurons list, store all added hidden neurons
        self.hidden_neurons = []

        # initialise hidden-output weight
        self.W_h_out = torch.randn(input_d + 1, output_d, device=device, dtype=dtype, requires_grad=True)
        first_neuron = HiddenNeuron(input_d)  # set first neuron
        self.hidden_neurons.append(first_neuron)
        self.first_train()  # the initial Casper model contains only one hidden neuron

        for old_hn_num in range(1, self.NUM_HIDDEN_NEURON):

            # create new neuron and add it into network
            new_h_neuron = HiddenNeuron(input_d + old_hn_num)

            # weight towards output layer for new neuron
            new_neuron_out_weight = torch.randn(1, output_d, device=device, dtype=dtype, requires_grad=True)

            # re-train the network
            self.train(new_h_neuron, new_neuron_out_weight)

            # print('Attention: ' + str(old_hn_num+1) + '   hidden neurons involved')

    def first_train(self):
        """
        Initially train a network with only one hidden neuron, no divided areas right now
        :return: None
        """
        # define loss function
        # time_period to check loss decrease and convergence
        time_period = int(15 + len(self.hidden_neurons) * self.P)
        loss_f = nn.CrossEntropyLoss()
        this_neuron_train_loss = []
        this_neuron_test_loss = []

        first_hidden = self.hidden_neurons[0]
        parameters = [first_hidden.w_in, self.W_h_out]

        optimizer = SARPROP(parameters, region_flag=0)

        i = 0
        while True:
            # forward
            coming_layer = self.train_x
            hn_out = first_hidden.compute_y_out(coming_layer)
            coming_layer = torch.cat((coming_layer, hn_out), 1)
            self.output_layer = torch.mm(coming_layer, self.W_h_out)

            train_loss = loss_f(self.output_layer, self.train_y)

            # determine convergence
            if (i % time_period == 0) and (len(this_neuron_train_loss) > 0):
                previous_loss = this_neuron_train_loss[-time_period]

                if train_loss < previous_loss:  # loss must decrease

                    delta = previous_loss - train_loss.item()  # always positive
                    if delta < 0.01 * previous_loss:
                        # print('first train converge at loop: ' + str(i))
                        break
                else:
                    # print('loss increase!!!')
                    break

            # record train loss
            this_neuron_train_loss.append(train_loss.item())

            # current test loss
            _, test_loss = self.get_test_output_loss()
            this_neuron_test_loss.append(test_loss)

            # backward
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # increment counter
            i += 1
        self.all_train_loss.append(this_neuron_train_loss)
        self.all_test_loss.append(this_neuron_test_loss)

    def train(self, new_neuron, new_neuron_out_weight):
        """
        Train whole network with the new hidden neuron, when converge, add the mature neuron into Casper network
        @param new_neuron: new hidden neuron
        @param new_neuron_out_weight:
        @return: None
        """

        # time_period to check loss decrease
        time_period = int(15 + len(self.hidden_neurons) * self.P)

        # define loss function
        loss_f = nn.CrossEntropyLoss()

        # define three different regions
        L1_region = [new_neuron.w_in]
        L2_region = [new_neuron_out_weight]
        L3_region = [self.W_h_out]

        for j, ne in enumerate(self.hidden_neurons):
            L3_region.append(ne.w_in)

        # for each train, reset the initial learning rage
        optimizer_l1 = SARPROP(L1_region, region_flag=0)
        optimizer_l2 = SARPROP(L2_region, region_flag=1)
        optimizer_l3 = SARPROP(L3_region, region_flag=2)

        i = 0  # loop counter
        this_hidden_train_loss_list = []
        this_hidden_test_loss_list = []
        while True:

            # forward
            self.forward(new_neuron, new_neuron_out_weight)

            # compute train loss
            this_epoch_train_loss = loss_f(self.output_layer, self.train_y)

            # determine convergence
            if (i % time_period == 0) and (len(this_hidden_train_loss_list) > 0):
                pre_loss = this_hidden_train_loss_list[-time_period]
                if this_epoch_train_loss < pre_loss:
                    delta = pre_loss - this_epoch_train_loss
                    if delta < 0.01 * pre_loss:
                        # print(str(i) + 'break')
                        break
                else:
                    # print('loss increase')
                    break

            # record train loss
            this_hidden_train_loss_list.append(this_epoch_train_loss)

            # record test loss
            _, this_epoch_test_loss = self.get_test_output_loss()
            this_hidden_test_loss_list.append(this_epoch_test_loss)

            # backward
            this_epoch_train_loss.backward()

            # update weight
            optimizer_l1.step()
            optimizer_l2.step()
            optimizer_l3.step()

            # clear gradients for next train
            optimizer_l1.zero_grad()
            optimizer_l2.zero_grad()
            optimizer_l3.zero_grad()

            i += 1

        self.all_train_loss.append(this_hidden_train_loss_list)
        self.all_test_loss.append(this_hidden_test_loss_list)

        # add trained neuron and update W_h_out
        self.hidden_neurons.append(new_neuron)
        new_W_h_out = torch.cat((self.W_h_out, new_neuron_out_weight))
        self.W_h_out = new_W_h_out.clone().detach().requires_grad_(True)  # construct new W_h_out

    def forward(self, new_neuron, new_neuron_out_weight):
        """
        Forward the network, hence get new output layer
        @return: None
        """
        # forward
        coming_layer = self.train_x

        for j, neuron in enumerate(self.hidden_neurons):
            current_out = neuron.compute_y_out(coming_layer)
            coming_layer = torch.cat((coming_layer, current_out), 1)

        # compute new neuron output
        new_neuron_out = torch.mm(new_neuron.compute_y_out(coming_layer), new_neuron_out_weight)

        # compute final output
        self.output_layer = torch.mm(coming_layer, self.W_h_out) + new_neuron_out

    def get_test_output_loss(self):
        """
        Use test data and current network (not necessary fully trained) to give last layer output and gives test loss
        @return: last layer output, current test loss

        """
        # define loss function
        loss_func = nn.CrossEntropyLoss()
        coming_layer = self.test_x

        for j, neuron in enumerate(self.hidden_neurons):
            current_out = neuron.compute_y_out(coming_layer)
            coming_layer = torch.cat((coming_layer, current_out), 1)

        current_test_output = torch.mm(coming_layer, self.W_h_out)
        current_test_loss = loss_func(current_test_output, self.test_y)

        return current_test_output, current_test_loss

    def model_eval(self):
        """
        After fully trained ,can be used to evaluate this model
        @return: precision, recall, F1, accuracy
        """
        test_output, _ = self.get_test_output_loss()
        pred_y = eval.predict_labels(test_output)
        combine = eval.combine_pred_real_labels(pred_y, self.test_y)
        eval_measures, accuracy = eval.evaluation(combine)
        print(eval_measures)
        print(accuracy)
        return eval_measures, accuracy

    def display_training_process(self):
        """
        Plot both training loss and testing loss, the title gives model id and the number of hidden units in network
        @return:
        """
        title = 'model id: ' + str(self.id) + ' num_hidden_neurons: ' + str(self.NUM_HIDDEN_NEURON)

        train_list = [item for sublist in self.all_train_loss for item in sublist]
        test_list = [item for sublist in self.all_test_loss for item in sublist]

        # mark when a new neuron was added
        timestamp = []
        timestamp_value = []
        critical_time = 0
        for i, li in enumerate(self.all_train_loss):
            if i < len(self.all_train_loss) - 1:
                critical_time += len(li)
                critical_val = self.all_train_loss[i + 1][0]
                timestamp.append(critical_time)
                timestamp_value.append(critical_val.item())

        # Plot training loss and testing loss
        plt.figure()
        plt.title(' training loss and testing loss \n ' + title)
        plt.xlabel('epoch')
        plt.ylabel('CrossEntropy loss')
        plt.plot(train_list, label='training loss')
        plt.plot(test_list, label='testing loss')
        plt.scatter(x=timestamp, y=timestamp_value, color='red', marker='x')
        plt.legend()
        plt.show()


# ##############################################################################################
# Run CasPer model on data and Evaluation
# ##############################################################################################

class CasPerModelComparison(object):
    def __init__(self,
                 data: torch.tensor,
                 use_lda: bool,
                 normalization_flag,
                 hidden_num=1
                 ):
        self.normalization_flag = normalization_flag
        self.train_data_list, self.test_data_list = dp_data.leave_one_participant_out(data, self.normalization_flag)

        self.NUM_MODEL = len(self.test_data_list)
        self.use_lda = use_lda

        self.NUM_HIDDEN = hidden_num

        # all train and test loss for 12 different models
        self.loss_12_train = []
        self.loss_12_test = []

        # all pred and real labes for 12 different models
        self.all_models_pred_labels = []
        self.all_models_real_labels = data[:, 0].long()

        self.train_models()

    def train_models(self):
        """
        Train 12 different model and collect information
        @return:
        """
        all_model_labels = []
        for m_id in range(self.NUM_MODEL):  # each loop trains a model

            train_data = self.train_data_list[m_id]
            test_data = self.test_data_list[m_id]

            # add bias layer
            train_data = data_pre.add_bias_layer(train_data)
            test_data = data_pre.add_bias_layer(test_data)

            # pre-processing of train and test data
            # LDA
            if self.use_lda:
                # use PCA before LDA to remove collinear variables
                train_data, test_data = data_pre.pca(train_data, test_data, 10)

                train_data = data_pre.lda_feature_selection(train_data, 3)
                test_data = data_pre.lda_feature_selection(test_data, 3)

            input_size = train_data.shape[1] - 1
            model = CasPerModel(input_size, 4, train_data=train_data, test_data=test_data,
                                num_hidden=self.NUM_HIDDEN, model_id=m_id)

            # store all losses for visualisation
            self.loss_12_train.append(model.all_train_loss)
            self.loss_12_test.append(model.all_test_loss)

            test_output, _ = model.get_test_output_loss()
            this_model_pred_label = eval.predict_labels(test_output)
            all_model_labels.append(this_model_pred_label)

            # print evaluation for a model
            print('------model ' + str(m_id))
            # model.model_eval()
            # model.display_training_process()

        self.all_models_pred_labels = torch.cat(all_model_labels)

    def final_evaluation(self):
        """
        This function gives overall evaluation of the 12 models.
        @return: first: measures for each class, second: average accuracy
        """
        combine = eval.combine_pred_real_labels(self.all_models_pred_labels, self.all_models_real_labels)
        eval_measures, overall_accuracy = eval.evaluation(combine)
        print(eval_measures)
        print(overall_accuracy)
        return eval_measures, overall_accuracy


