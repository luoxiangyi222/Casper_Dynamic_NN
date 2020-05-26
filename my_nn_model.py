# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import evaluation as eval
import depression_data as dp_data
import numpy as np


# ##############################################################################################
# feed forward nn models
# ##############################################################################################

class OneHiddenNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OneHiddenNN, self).__init__()
        self.ih = nn.Linear(input_size, hidden_size, bias=True)
        self.ho = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x = torch.sigmoid(self.ih(x))
        output = self.ho(x)
        return output


class FFNNModelComparison(object):
    def __init__(self,
                 data: torch.tensor,
                 use_lda: bool,
                 # use_GA_feature_selection,
                 learning_rate,
                 normalization_flag,
                 epochs,
                 hidden_num=10
                 ):
        self.data = data

        # leave one participates out
        self.train_data_list, self.test_data_list = dp_data.leave_one_participant_out(data, self.normalization_flag)

        # record all models pred and real labels
        self.all_real_ys = data[:, 0].long()
        self.all_final_pred_ys = []

        self.use_lda = use_lda
        self.lr = learning_rate
        self.normalization_flag = normalization_flag
        self.hidden_num = hidden_num
        self.epochs = epochs

        self.num_participants = 12

        # record loss and accuracy for visualization
        self.train_loss_12 = []
        self.test_loss_12 = []
        self.train_accuracy_12 = []
        self.test_accuracy_12 = []

        # train 12 different models
        self.train()
        self.all_final_pred_ys = torch.cat(self.all_final_pred_ys)

    def check_converge(self):
        pass

    def train(self):

        for p in range(self.num_participants):
            train_data = self.train_data_list[p]
            test_data = self.test_data_list[p]

            input_size = train_data.shape[1] - 1

            train_X = train_data[:, 1:]
            train_Y = train_data[:, 0].long()
            test_X = test_data[:, 1:]
            test_Y = test_data[:, 0].long()

            # for every iteration in cross validation, build new net
            net = OneHiddenNN(input_size, self.hidden_num, 4)
            loss_func = nn.CrossEntropyLoss()
            optimiser = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0.01)

            # store all losses for visualisation
            model_train_loss = []
            model_test_loss = []
            model_train_accuracy = []
            model_test_accuracy = []

            for i in range(self.epochs):
                # train
                train_pred_output = net(train_X)
                # record train loss and accuracy
                train_loss = loss_func(train_pred_output, train_Y)
                model_train_loss.append(train_loss.item())
                train_pred_label = eval.predict_labels(train_pred_output)
                _, current_train_accuracy = eval.train_evaluation(train_pred_label, train_Y)
                model_train_accuracy.append(current_train_accuracy)


                # test
                test_pred_output = net(test_X)
                # record test loss and accuracy
                test_loss = loss_func(test_pred_output, test_Y)
                model_test_loss.append(test_loss.item())
                test_pred_label = eval.predict_labels(test_pred_output)
                _, current_test_accuracy = eval.train_evaluation(test_pred_label, test_Y)
                model_test_accuracy.append(current_test_accuracy)

                # back prop !!!!!
                # clear gradients for next train
                optimiser.zero_grad()
                # perform backward pass
                train_loss.backward()
                # call the step function on an Optimiser makes an update to its
                # parameters
                optimiser.step()

            self.train_loss_12.append(model_train_loss)
            self.test_loss_12.append(model_test_loss)
            self.train_accuracy_12.append(model_train_accuracy)
            self.test_accuracy_12.append(model_test_accuracy)

            print('Now predicting testing set:')

            # record final results of the model
            prediction_test = net(test_X)
            final_model_pred_label = eval.predict_labels(prediction_test)
            test_loss = loss_func(prediction_test, test_Y)
            self.all_final_pred_ys.append(final_model_pred_label)

            print('truth y')
            print(test_Y)
            print('pred y')
            print(final_model_pred_label)
            print('The loss for testing set is: ' + str(test_loss))

    def final_evaluation(self):
        combine = eval.combine_pred_real_labels(self.all_final_pred_ys, self.all_real_ys)
        eval_measures, overall_accuracy = eval.evaluation(combine)
        print('evaluation for all model')
        print(eval_measures)
        print(overall_accuracy)
        return eval_measures, overall_accuracy

    def visualization(self):
        for i in range(self.num_participants):
            train_loss = self.train_loss_12[i]
            test_loss = self.test_loss_12[i]
            train_accuracy = self.train_accuracy_12[i]
            test_accuracy = self.test_accuracy_12[i]

            # Plot training loss
            fig = plt.figure(figsize=(8, 10))
            a0 = fig.add_subplot(211)
            a1 = fig.add_subplot(212)

            # display training loss and testing loss
            a0.set_title('training loss and testing loss during training')
            a0.set_xlabel('epoch')
            a0.set_ylabel('CrossEntropy loss')
            a0.plot(train_loss, label='train loss')
            a0.plot(test_loss, label='test loss')
            a0.legend()

            # display training accuracy and testing accuracy
            a1.set_title('accuracy during training')
            a1.set_xlabel('epoch')
            a1.set_ylabel('accuracy')
            a1.plot(train_accuracy, label='train accuracy')
            a1.plot(test_accuracy, label='test accuracy')
            a1.legend()
            plt.show()

