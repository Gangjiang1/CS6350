import sys
import AdaBoost
import Bagging
import RandomForest
from Metrics import get_accuracy
from Plotter import plot_classifier


## evaluate algorithm
def evaluate(train_dir, test_dir, algorithm, *args):
    predictions = algorithm(train_dir, test_dir, *args)
    test = AdaBoost.load_csv(test_dir)
    actual = test['y'].to_list()
    error = 1 - get_accuracy(actual, predictions)
    return error


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    alg = sys.argv[3]
    t_max = int(sys.argv[4])
    try:
        max_samples = int(sys.argv[5])
        max_features = int(sys.argv[6])
    except IndexError:
        max_samples = 500
        max_features = 3

    train_errors = []
    test_errors = []
    iterations = list(range(1, t_max+1))
    for t in iterations:
        print(f'Building {alg} with max iteration = {t}')
        if alg == 'adaboost':
            algorithm = AdaBoost.adaboost
            print('Evaluating training error...')
            train_error = evaluate(train_file, train_file, algorithm, t)
            print('Evaluating test error...')
            test_error = evaluate(train_file, test_file, algorithm, t)
        elif alg == 'bagging':
            algorithm = Bagging.bagging
            print('Evaluating training error...')
            train_error = evaluate(train_file, train_file, algorithm, t, max_samples)
            print('Evaluating test error...')
            test_error = evaluate(train_file, test_file, algorithm, t, max_samples)
        elif alg == 'randomforest':
            algorithm = RandomForest.randomforest
            print('Evaluating training error...')
            train_error = evaluate(train_file, train_file, algorithm, t, max_samples, max_features)
            print('Evaluating test error...')
            test_error = evaluate(train_file, test_file, algorithm, t, max_samples, max_features)
        train_errors.append(train_error)
        test_errors.append(test_error)
    plot_sel2 = input('Do you want to plot error trajectories? (Y/N)')
    if plot_sel2 == 'Y':
        plot_classifier(iterations, train_errors, test_errors)
