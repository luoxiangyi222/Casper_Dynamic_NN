# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import evaluation as eval
import depression_data as dp_data
import numpy as np
import data_preprocessing as data_pre


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
        self.use_lda = use_lda
        self.lr = learning_rate
        self.normalization_flag = normalization_flag
        self.NUM_HIDDEN = hidden_num
        self.epochs = epochs

        self.time_period = 50

        # leave one participates out
        self.train_data_list, self.test_data_list = dp_data.leave_one_participant_out(data, self.normalization_flag)
        self.NUM_MODEL = len(self.test_data_list)
        
        # define input and output size
        self.output_size = 4

        # record all models pred and real labels
        self.all_real_label = data[:, 0].long()
        self.all_final_pred_label = []

        # record loss and accuracy for visualization
        self.train_loss_12 = []
        self.test_loss_12 = []
        # self.train_accuracy_12 = []
        # self.test_accuracy_12 = []

        # train 12 different models
        self.train_models()
        # self.all_final_pred_label = torch.cat(self.all_final_pred_label)

    def train_models(self):
        """
        Train 12 different models
        @return: 
        """

        for m_id in range(self.NUM_MODEL):
            train_data = self.train_data_list[m_id]
            test_data = self.test_data_list[m_id]

            # pre-processing of train and test data
            # LDA
            if self.use_lda:
                # use PCA before LDA to remove collinear variables
                train_data, test_data = data_pre.pca(train_data, test_data, 10)

                train_data = data_pre.lda_feature_selection(train_data, 3)
                test_data = data_pre.lda_feature_selection(test_data, 3)
                
            train_X = train_data[:, 1:]
            train_Y = train_data[:, 0]
            
            test_X = test_data[:, 1:]
            test_Y = test_data[:, 0]

            input_size = train_data.shape[1] - 1

            # build new net
            net = OneHiddenNN(input_size, self.NUM_HIDDEN, self.output_size)
            loss_func = nn.CrossEntropyLoss()
            optimiser = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=0.01)

            # store all losses for visualisation
            model_train_loss = []
            model_test_loss = []

            for i in range(self.epochs):
                # train
                train_pred_output = net(train_X)
                # record train loss and accuracy
                this_epoch_train_loss = loss_func(train_pred_output, train_Y)
                model_train_loss.append(this_epoch_train_loss.item())

                # determine convergence
                if (i % self.time_period == 0) and (len(model_train_loss) > 0):
                    pre_loss = model_train_loss[-self.time_period]
                    if this_epoch_train_loss.item() < pre_loss:
                        delta = pre_loss - this_epoch_train_loss.item()
                        if delta < 0.01 * pre_loss:
                            print(str(i) + 'converge')
                            break
                    else:
                        print('loss increase')
                        break

                # test
                test_pred_output = net(test_X)
                # record test loss and accuracy
                test_loss = loss_func(test_pred_output, test_Y)
                model_test_loss.append(test_loss.item())

                # back prop
                # clear gradients for next train
                optimiser.zero_grad()
                # perform backward pass
                this_epoch_train_loss.backward()
                # call the step function on an Optimiser makes an update to its parameters
                optimiser.step()

            self.train_loss_12.append(model_train_loss)
            self.test_loss_12.append(model_test_loss)

            print('Now predicting testing set:')

            # record final results of the model
            test_last_layer = net(test_X)
            final_model_pred_label = eval.predict_labels(test_last_layer)
            self.all_final_pred_label.append(final_model_pred_label)

    def final_evaluation(self):
        combine = eval.combine_pred_real_labels(self.all_final_pred_label, self.all_real_label)
        eval_measures, overall_accuracy = eval.evaluation(combine)
        print('evaluation for all model')
        print(eval_measures)
        print(overall_accuracy)
        return eval_measures, overall_accuracy

    def visualization(self):
        for i in range(self.NUM_MODEL):
            train_loss = self.train_loss_12[i]
            test_loss = self.test_loss_12[i]

            # Plot training loss
            plt.figure()

            # display training loss and testing loss
            plt.title('training loss and testing loss during training')
            plt.xlabel('epoch')
            plt.ylabel('CrossEntropy loss')
            plt.plot(train_loss, label='train loss')
            plt.plot(test_loss, label='test loss')
            plt.legend()
            plt.show()

