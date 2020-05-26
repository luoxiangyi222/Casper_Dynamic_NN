# comp4660 assignment 1 code
# Author: Xiangyi Luo (u6162693)
# Time: May 2020


import numpy as np
import torch
import pdb


def train_evaluation(current_pred_ys, current_real_ys):
    """
    Used to recording evaluation of model during training process
    """
    combine = combine_pred_real_labels(current_pred_ys, current_real_ys)
    eval_measures, accuracy = evaluation(combine)
    return eval_measures, accuracy


def combine_pred_real_labels(pred_ys, real_ys):
    pred_ys = pred_ys.unsqueeze(dim=1)
    real_ys = real_ys.unsqueeze(dim=1)
    pred_real_ys = torch.cat([pred_ys, real_ys], dim=1)
    return pred_real_ys


def predict_labels(out: torch.tensor):
    """
    :param out: output of last layers
    :return: predict labels
    """
    soft_max_out = torch.softmax(out, 1)
    preds = torch.argmax(soft_max_out, dim=1)
    return preds


def compute_evaluation_measure(n_correct, tp_fp, tp_fn):

    if n_correct == 0:
        return 0, 0, 0
    else:
        precision = n_correct / tp_fp
        recall = n_correct/tp_fn
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluation(pred_real_y: torch.tensor):
    """
    :param pred_real_y: given pairs of pred label and real label
    :return: eval_measures, an array contains precision, recall and f1 for each class, the last row is average values
    also return the overall accuracy
    """
    pred_real_y = np.array(pred_real_y)
    eval_measures = []

    match = pred_real_y[pred_real_y[:, 0] == pred_real_y[:, 1]]

    for label in range(4):
        correct = len(match[match[:, 0] == label])
        tp_fp = len(pred_real_y[pred_real_y[:, 0] == label])
        tp_fn = len(pred_real_y[pred_real_y[:, 1] == label])
        eval_measures.append((compute_evaluation_measure(correct, tp_fp, tp_fn)))

    eval_measures = np.array(eval_measures)
    average = np.sum(eval_measures, axis=0) / 4
    eval_measures = np.vstack([eval_measures, average])

    # overall accuracy
    N = len(pred_real_y)
    overall_accuracy = len(match) / N
    return eval_measures, overall_accuracy
