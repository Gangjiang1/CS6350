import SVM
import KernelSVM
import sys


## calculate error
def get_error(predicted, actual):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return 1 - correct/float(len(actual))

## Driver to run the algorithm, and return list of predicted values and error rates
def run(train_file, test_file, algorithm, gamma, C, max_iter, kernel, which, schedule_a):
    train_x, train_y = algorithm.load_csv(train_file)
    test_x, test_y = algorithm.load_csv(test_file)

    if algorithm == SVM:
        svm = algorithm.SVM(C=C, step=gamma, max_iter=max_iter, schedule_a=schedule_a, which=which)
        svm.fit(train_x, train_y)
        train_preds = svm.predict(train_x)
        test_preds = svm.predict(test_x)
        train_truths = train_y['Outcome']
        test_truths = test_y['Outcome']

    elif algorithm == KernelSVM:
        if kernel == 'linear':
            kern = algorithm.linear
        elif kernel == 'gaussian':
            kern = algorithm.gaussian
        else:
            raise NameError('Unknown kernel function')
        kernelsvm = algorithm.KernelSVM(C=C, gamma=gamma, kernel=kern, max_iter=max_iter)
        kernelsvm.fit(train_x, train_y)
        train_preds = kernelsvm.predict(train_x)
        test_preds = kernelsvm.predict(test_x)
        train_truths = train_y
        test_truths = test_y

    else:
        raise NameError('Unknown algorithm')

    train_error = get_error(train_preds, train_truths)
    test_error = get_error(test_preds, test_truths)
    print(f'Training error is {train_error}')
    print(f'Test error is {test_error}')
    return


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    alg = sys.argv[3]
    step = float(sys.argv[4])
    C = float(sys.argv[5])
    max_iter = int(sys.argv[6])
    try:
        kernel = sys.argv[7]
        which_schedule = int(sys.argv[8])
        schedule_a = float(sys.argv[9])
    except IndexError:
        kernel = 'linear'
        which_schedule = 1
        schedule_a = 0.5

    if alg == 'svm':
        algorithm = SVM
    elif alg == 'kernel_svm':
        algorithm = KernelSVM
    else:
        raise NameError('Unknown algorithm')
    run(train_file, test_file, algorithm, step, C, max_iter, kernel, which_schedule, schedule_a)
