import pandas as pd
import numpy as np
from Plotter import plot_regressor
import sys


## load a csv file
def load_csv(filepath):
    features = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr']
    outcome = ['Slump']
    dataset = pd.read_csv(filepath, names=features+outcome)
    x = dataset[features]
    y = dataset[outcome]
    return x, y

## define cost function
def cost(x, y, w):
    m = y.shape[0]
    prediction = np.dot(x, w)
    error = prediction - y['Slump']
    j = 1 / (2 * m) * np.dot(error.T, error)
    return error, j

## build gradient descent algorithm
def gradient_descent(x, y, alpha, tolerance, max_iter):
    m = y.shape[0]
    w = [0] * x.shape[1]
    past_j = []
    past_w = [w]
    converge = 999
    iteration = 1
    while converge >= tolerance:
        if iteration >= max_iter:
            print('Warning: Max iteration reached. Consider adjusting learning rate or increase max iteration')
            break
        error, j = cost(x, y, w)
        past_j.append(j)
        gradient = alpha * (1/m) * np.dot(x.T, error)
        w = w - gradient
        past_w.append(w)
        converge = np.sqrt(np.dot(gradient.T, gradient))
        iteration += 1
    return past_w, past_j, iteration

## build stochastic gradient descent algorithm
def stochastic_gradient_descent(x, y, alpha, tolerance, max_iter):
    m = y.shape[0]
    w = [0] * x.shape[1]
    past_j = []
    past_w = [w]
    iteration = 1
    for _ in range(max_iter):
        rand_ind = np.random.choice(range(m), m, replace=False)
        for ind in rand_ind:
            error, j = cost(x, y, w)
            past_j.append(j)
            gradient = alpha * (1/m) * np.dot(x.iloc[ind].T, error[ind])
            w = w - gradient
            past_w.append(w)
            converge = np.sqrt(np.dot(gradient.T, gradient))
            if converge < tolerance:
                break
        iteration += 1
    return past_w, past_j, iteration

## build driver to run algorithm and return analyzed results
def driver(train_x, train_y, test_x, test_y, r, tol, max_iteration, algorithm):
    w_list, j_list, iteration = algorithm(train_x, train_y, r, tol, max_iteration)

    ## adjust learning rate
    while any(j >= 100 for j in j_list):
        print('Cost is increasing. Decreasing learning rate and trying again...')
        r = r / 2
        w_list, j_list = algorithm(train_x, train_y, r, tol, max_iteration)
    if iteration >= max_iteration:
        print('Warning: Max iteration reached. Consider adjusting learning rate or increase max iteration')
    final_w = w_list[-1]
    test_error, test_j = cost(test_x, test_y, final_w)
    print(f'Stopped at iteration {iteration}')
    print(f'Cost of the test set is {round(test_j, 3)}')
    print(f'Coefficients are {final_w}')
    print(f'Learning rate is {r}')
    plot_sel = input('Do you want to plot cost trajectory? (Y/N)')
    if plot_sel == 'Y':
        plot_regressor(range(len(j_list)), j_list)
    return


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    alg = sys.argv[3]
    train_x, train_y = load_csv(train_file)
    test_x, test_y = load_csv(test_file)
    try:
        r = float(sys.argv[4])
        tol = float(sys.argv[5])
        max_iteration = int(sys.argv[6])
    except IndexError:
        r = 1.0
        tol = 1e-6
        max_iteration = 20000

    print(f'Using {alg}...')
    if alg == 'gradient_descent':
        driver(train_x, train_y, test_x, test_y, r, tol, max_iteration, gradient_descent)
    if alg == 'stochastic_gradient_descent':
        driver(train_x, train_y, test_x, test_y, r, tol, max_iteration, stochastic_gradient_descent)
