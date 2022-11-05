import Perceptron
import VotedPerceptron
import AveragedPerceptron
import sys


## calculate error
def get_error(predicted, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return 1 - correct/float(len(actual))

## Driver to run the algorithm, and return list of predicted values and error rates
def run(train_file, test_file, algorithm, alpha, max_iter):
    train_x, train_y = algorithm.load_csv(train_file)
    test_x, test_y = algorithm.load_csv(test_file)
    predicted_list = []

    w = algorithm.learn(train_x, train_y, alpha, max_iter)
    for i in range(test_y.shape[0]):
        predicted = algorithm.predict(test_x.iloc[i].T, w)
        predicted_list.append(predicted)
    test_error = get_error(predicted_list, test_y['Outcome'])
    print(f'Prediction error is {test_error}')
    return


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    alg = sys.argv[3]
    try:
        alpha = sys.argv[4]
        max_iter = sys.argv[5]
    except IndexError:
        alpha = 0.5
        max_iter = 10
    if alg == 'perceptron':
        algorithm = Perceptron
    elif alg == 'voted_perceptron':
        algorithm = VotedPerceptron
    elif alg == 'averaged_perceptron':
        algorithm = AveragedPerceptron
    else:
        raise NameError('Unknown algorithm. Please check your spelling.')
    run(train_file, test_file, algorithm, alpha, max_iter)
