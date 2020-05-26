# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020


import torch
import torch.nn as nn
import data_preprocessing
import matplotlib.pyplot as plt
import evaluation


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

        # parameter
        self.k1 = 1e-7
        self.T = 0.01
        self.state = {}
        for i in range(len(params)):
            self.state[i] = {}
        self.step_counter = 0

    def zero_grad(self):
        # set all grad to zero
        if self.params:
            for par_id, p in enumerate(self.params):
                p.grad.zero_()

    def step(self):
        # need to first generate a random new point in the space

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

            self.step_counter += 1

            # compute SA gradient
            sa_term = self.k1 * (2 ** (- self.T * self.step_counter)) * (p.sign()) * p * p

            SA_grad = grad - sa_term
            sign = SA_grad.mul(state['prev_grad']).sign()

            # set factor according to sign
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1

            # update stepsizes with step size updates
            step_size.mul_(sign).clamp_(step_size_min, step_size_max)

            # for dir<0, dfdx=0
            # for dir>=0 dfdx=dfdx
            SA_grad = SA_grad.clone(memory_format=torch.preserve_format)
            SA_grad[sign.eq(etaminus)] = 0

            # update parameters
            with torch.no_grad():
                p.addcmul_(SA_grad.sign(), step_size, value=-1)

            # save previous gradient
            state['prev_grad'] = SA_grad.clone().detach()


class HiddenNeuron(object):
    def __init__(self,
                 num_in: int,
                 ):
        # randomly initialise the come and away weights
        self.w_in = torch.randn(num_in, 1, device=device, dtype=dtype, requires_grad=True)

        self.y_in = None
        self.y_out = None

    def compute_y_in(self, input_layer):
        self.y_in = torch.mm(input_layer, self.w_in)

    def compute_y_out(self, input_layer):
        self.compute_y_in(input_layer)
        self.y_out = torch.tanh(self.y_in)
        return self.y_out


class CasPerModel(object):
    def __init__(self,
                 input_d: int,
                 output_d: int,
                 train_data: torch.Tensor,
                 test_data: torch.Tensor,
                 max_hidden_num
                 ):
        # input, output dimension
        self.input_d = input_d
        self.output_d = output_d

        # hyper parameter
        self.MAX_HIDDEN_NEURON_NUM = max_hidden_num
        self.P = 1

        # store all data and test data, first column is label
        self.train_x = train_data[:, 1:]
        self.train_y = train_data[:, 0].long()
        self.test_x = test_data[:, 1:]
        self.test_y = test_data[:, 0].long()

        self.all_train_loss = []
        self.all_pred_loss = []

        # neurons list, store all added hidden neurons
        self.hidden_neurons = []

        # initialise hidden-output weight
        self.W_h_out = torch.randn(input_d + 1, output_d, device=device, dtype=dtype, requires_grad=True)
        first_neuron = HiddenNeuron(input_d)  # set first neuron
        self.hidden_neurons.append(first_neuron)
        self.first_train()

        for old_hn_num in range(1, self.MAX_HIDDEN_NEURON_NUM):
            print(str(old_hn_num) + '      hidden neurons involved')

            # create new neuron and add it into network
            new_h_neuron = HiddenNeuron(input_d + old_hn_num)

            # weight towards output layer for new neuron
            new_neuron_out_weight = torch.randn(1, output_d, device=device, dtype=dtype, requires_grad=True)

            # re-train the network
            self.train(new_h_neuron, new_neuron_out_weight)

    def first_train(self):
        """
        Initially train a network without any hidden neuron
        :param epoch: default 20
        :return: None
        """
        # define loss function

        # time_period to check loss decrease
        time_period = int(15 + len(self.hidden_neurons) * self.P)
        loss_f = nn.CrossEntropyLoss()
        all_loss = []

        first_n = self.hidden_neurons[0]
        parameters = [self.W_h_out, first_n.w_in]
        optimizer = SARPROP(parameters, region_flag=0)

        i = 0
        while True:
            # forward
            # print(str(i)+'        ||||||||||||')
            coming_layer = self.train_x
            hn_out = first_n.compute_y_out(coming_layer)
            coming_layer = torch.cat((coming_layer, hn_out), 1)
            self.train_pred_y = torch.mm(coming_layer, self.W_h_out)

            loss = loss_f(self.train_pred_y, self.train_y)

            # determine convergence
            if (i % time_period == 0) and (len(all_loss) > 0):
                pre_loss = all_loss[-time_period]
                if loss < pre_loss:
                    delta = pre_loss - loss
                    if delta < 0.01 * pre_loss:
                        # print(str(i) + '      break')
                        break
                else:
                    break

            # record loss value
            all_loss.append(loss)
            self.all_train_loss.append(loss)

            # current pred loss
            _, pred_loss = self.predict()
            self.all_pred_loss.append(pred_loss)

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # increment counter
            i += 1

    def train(self, new_neuron, new_neuron_out_weight):

        # time_period to check loss decrease
        time_period = int(15 + len(self.hidden_neurons) * self.P)

        # define loss function
        loss_f = nn.CrossEntropyLoss()
        all_loss_for_this_train = []

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
        while True:

            # print('========= train round ' + str(i) + '===============')

            # forward
            coming_layer = self.train_x

            for j, neuron in enumerate(self.hidden_neurons):
                current_out = neuron.compute_y_out(coming_layer)
                coming_layer = torch.cat((coming_layer, current_out), 1)

            # compute new neuron output
            new_neuron_out = torch.mm(new_neuron.compute_y_out(coming_layer), new_neuron_out_weight)

            # compute final output
            self.train_pred_y = torch.mm(coming_layer, self.W_h_out) + new_neuron_out

            loss = loss_f(self.train_pred_y, self.train_y)

            # determine convergence
            if (i % time_period == 0) and (len(all_loss_for_this_train) > 0):
                pre_loss = all_loss_for_this_train[-time_period]
                if loss < pre_loss:
                    delta = pre_loss - loss
                    if delta < 0.01 * pre_loss:
                        # print(str(i) + '      break')
                        break
                else:
                    break

            # record train loss
            all_loss_for_this_train.append(loss)
            self.all_train_loss.append(loss)
            # record test loss
            _, pred_loss = self.predict()
            self.all_pred_loss.append(pred_loss)

            # backward
            loss.backward()

            # update weight
            optimizer_l1.step()
            optimizer_l2.step()
            optimizer_l3.step()

            # clear gradients for next train
            optimizer_l1.zero_grad()
            optimizer_l2.zero_grad()
            optimizer_l3.zero_grad()

            i += 1

        # add trained neuron and update W_h_out
        self.hidden_neurons.append(new_neuron)
        new_W_h_out = torch.cat((self.W_h_out, new_neuron_out_weight))
        self.W_h_out = new_W_h_out.clone().detach().requires_grad_(True)  # construct new W_h_out

    def predict(self):
        """
        :param test_data: testing set
        :return: pred labels and loss
        """

        # forward
        coming_layer = self.test_x

        for j, neuron in enumerate(self.hidden_neurons):
            current_out = neuron.compute_y_out(coming_layer)
            coming_layer = torch.cat((coming_layer, current_out), 1)

        test_pred_y = torch.mm(coming_layer, self.W_h_out)

        loss_func = nn.CrossEntropyLoss()
        test_loss = loss_func(test_pred_y, self.test_y)
        return test_pred_y, test_loss


# ##############################################################################################
# Run CasPer model on data and Evaluation
# ##############################################################################################



class CasPerModelComparison(object):
    def __init__(self,
                 data: torch.tensor,
                 use_lda: bool,
                 normalization_flag,
                 hidden_num=10
                 ):
        self.data = data
        self.use_lda = use_lda
        self.normalization_flag = normalization_flag

        self.num_participants = 12
        self.pred_ys = []

        self.loss_12_train = []
        self.loss_12_test = []

        self.all_pred_labels = []
        self.real_ys = data[:, 0].long()

        self.train()

    def train(self):
        pred_all_labels = []
        for test_p in range(self.num_participants):
            print(str(test_p) + '============================================')
            start = 16 * test_p
            end = start + 16

            train_data = torch.cat([self.data[:start], self.data[end:]])
            test_data = self.data[start:end]

            # pre-processing of train and test data
            # LDA
            if self.use_lda:
                train_data = data_preprocessing.remove_colinear_features(train_data, 0.9)
                train_data = data_preprocessing.lda_feature_selection(train_data, 3)
                test_data = data_preprocessing.remove_colinear_features(test_data, 0.9)
                test_data = data_preprocessing.lda_feature_selection(test_data, 3)
            # Normalization
            train_data = data_preprocessing.normalization(train_data, self.normalization_flag)
            test_data = data_preprocessing.normalization(test_data, self.normalization_flag)

            # add bias
            train_data = data_preprocessing.add_bias_layer(train_data)
            test_data = data_preprocessing.add_bias_layer(test_data)
            input_size = train_data.shape[1] - 1

            model = CasPerModel(input_size, 4, train_data=train_data, test_data=test_data, max_hidden_num=10)

            # store all losses for visualisation
            self.loss_12_train.append(model.all_train_loss)
            self.loss_12_test.append(model.all_pred_loss)

            test_pred_output, test_loss = model.predict()
            pred_label_for_this_train = evaluation.predict_labels(test_pred_output)
            pred_all_labels.append(pred_label_for_this_train)

            print('The loss for testing set is: ' + str(test_loss))

        self.all_pred_labels = torch.cat(pred_all_labels)


    def evaluation(self):

        combine = evaluation.combine_pred_real_labels(self.all_pred_labels, self.real_ys)
        eval_measures, overall_accuracy = evaluation.evaluation(combine)
        print(eval_measures)
        print(overall_accuracy)
        return eval_measures, overall_accuracy

    def visualization(self):
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(self.loss_12_train)
        print(self.loss_12_test)
        for i in range(self.num_participants):
            all_train_loss = self.loss_12_train[i]
            all_eval_loss = self.loss_12_test[i]

            # Plot training loss
            plt.figure()
            plt.title(' training loss and testing loss during training')
            plt.xlabel('epoch')
            plt.ylabel('CrossEntropy loss')
            plt.plot(all_train_loss, label='training loss')
            plt.plot(all_eval_loss, label='testing loss')
            plt.legend()
            plt.show()


