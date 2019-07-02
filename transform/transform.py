__author__ = 'Horace'

import os, sys, setting
import numpy, pandas


def group_class_variable(dataset, column_index, target_list=list(), replace=False):
    column_name = dataset.columns[column_index-1]
    column = dataset[dataset.columns[column_index-1]]
    for value in column:
        if value not in target_list:
            value = "Others"

    dataset[str(column_name + "_class")] = pandas.Series(["Others" if col not in target_list else col for col in column])
    if replace == True:
        dataset = dataset.drop(column_name, axis=1)
    return dataset

def transform_binary_col(dataset, column_index, replace=False):
    column_name = dataset.columns[column_index]
    column = dataset[dataset.columns[column_index]]
    column_list = list(set(column))
    if len(column_list) != 2:
        raise Exception("Column: " + column_name + " does not contain binary values")

    dataset[str(column_name + "_num")] = pandas.Series([column_list.index(col)+1 for col in column])
    if replace == True:
        dataset = dataset.drop(column_name, axis=1)
    return column_list, dataset

def transform_nominal_col(dataset, column_index, replace=False):
    column_name = dataset.columns[column_index-1]
    column = dataset[dataset.columns[column_index-1]]
    column_list = list(set(column))
    if len(column_list) <= 2:
        raise Exception("Column: " + column_name + " does not contain nominal values")

    dataset[str(column_name + "_num")] = pandas.Series([column_list.index(col)+1 for col in column])
    if replace == True:
        dataset = dataset.drop(column_name, axis=1)
    return column_list, dataset

def convert_to_datetime(raw_data, column_index, new_column_name, drop=False):
    raw_data[new_column_name] = pandas.Series([pandas.to_datetime(element)
                                               for element in raw_data[raw_data.columns[column_index]]])
    if drop is True:
        raw_data.drop(raw_data.columns[column_index], axis=1)

    print raw_data
    return raw_data


def load_vocab_list(dataset):
    vocab_set = set([])
    for obs in dataset:
        vocab_set = vocab_set | set(obs)

    return list(vocab_set)


def word_to_vector(vocab_list, input_list):
    vector = [0]*len(vocab_list)
    for word in input_list:
        if word in vocab_list:
            vector[vocab_list.index(word)] += 1
        else:
            print "The word: %s is not in vocabulary list." % word

    return vector


def word_to_matrix(vocab_list, input_list):
    vector_list = []
    for word in input_list:
        vector_list.append(word_to_vector(vocab_list, word))

    return vector_list
