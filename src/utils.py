import numpy as np
import pickle
import os
import random
from collections import Counter
from typing import Optional, Iterable, TypeVar
import torch

def print_timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("done in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def get_steps(len_dataset, n_samples):
    l = len_dataset
    samples = l/n_samples
    array = np.zeros((int(samples), n_samples), dtype=int)
    row = 0
    col = 0
    for i in range(len_dataset):
        array[row, col] = i+1
        col +=1
        if (i+1)%5==0: 
            row +=1
            col = 0
    return array

class TrackF1Score():
    def __init__(self, path):
        """
        save the f1 scores in a single pickle-file
        """
        self.path = path
        self.f1_score = {}
        self.f1_score['entropy_thresholds'] = {}
        self.best_f1_score = (0,0)

    def add_f1_score(self, entr_thresh, score):
        self.f1_score['entropy_thresholds'][f'{entr_thresh}'] = score
        if score > self.best_f1_score[1]:
            self.best_f1_score = (np.round(entr_thresh, 2), score)
    def save_pickle(self, name_type):
        self.f1_score[f'best_f1_score'] = {
                'threshold':    self.best_f1_score[0],
                'f1_score':     self.best_f1_score[1]
        }
        with open(os.path.join(self.path, f'f1_scores_{name_type}_meta.pkl'), 'wb') as f:
            pickle.dump(self.f1_score, f)

class MeanF1Score():
    def __init__(self, path):
        """
        save the mean of f1 scores
        """
        self.path = path
        self.f1_score = []

    def add_f1_score(self, score):
        self.f1_score.append(score)
        
    def mean_f1_score(self):
        f1_score = np.round(np.mean(self.f1_score), 2)
        print('F1 score: ', f1_score)
        with open(os.path.join(self.path, f'f1_score.pkl'), 'wb') as f:
            pickle.dump(f1_score, f)
        with open(os.path.join(self.path, 'metrics.txt'), 'a') as file:  
            file.write("F1 score: {:6.2f} %/n".format(f1_score))
        

def random_split(dataset, total_num_sequences, cfg):
    train_sequences = list(range(1, total_num_sequences+1))
    val_sequences = []
    for i in range(int(np.round(cfg.crossvalidation.val_size*total_num_sequences, decimals=0))):
        choice = random.choice(train_sequences)
        val_sequences.append(choice)
        train_sequences.remove(choice)
    return train_sequences, val_sequences

def counts_array_to_data_list(counts_array, max_size=None):
    if max_size is None:
        max_size = np.sum(counts_array)  # max of counted array entry
    if np.sum(counts_array) != 0: 
        counts_array = (counts_array / np.sum(counts_array) * max_size).astype("uint32")
    counts_dict = {}
    for i in range(1, len(counts_array) + 1):
        counts_dict[i] = counts_array[i - 1]
    return list(Counter(counts_dict).elements())

def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)

def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Iterable[T] = None,
    custom_msg: Optional[str] = None,
    ) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value
