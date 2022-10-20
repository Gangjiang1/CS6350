## import packages
import numpy as np

## calculate weighted error
def get_error(weights, actual, predicted):
    error = 0.0
    for i in range(len(actual)):
        if actual[i] != predicted[i]:
            error += weights[i]
    return error

## calculate accuracy
def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))

## metric selector
def get_metric(props, metric):
    if metric == 'entropy':
        entropy = sum([-v * np.log(v) for v in props if v != 0])
        return entropy
    if metric == 'majority_error':
        me = 1-max(props)
        return me
    if metric == 'gini':
        gini = 1-sum([v ** 2 for v in props])
        return gini
