import pandas as pd
import numpy as np


## load csv files
def load_csv(filepath):
    features = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    outcome = ['Outcome']
    dataset = pd.read_csv(filepath, names=features + outcome)
    x = dataset[features]
    y = dataset[outcome]
    x['bias'] = 1
    y.loc[y['Outcome'] == 0, 'Outcome'] = -1
    return x, y

class SVM:

    def __init__(self, C, step, max_iter, schedule_a, which):
        self.w = None
        self.C = C
        self.step = step
        self.max_iter = max_iter
        self.schedule_a = schedule_a
        self.which_schedule = which

    ## Build perceptron algorithm
    def fit(self, X, y):
        print('Currently using primal SVM...')
        np.random.seed(123)
        self.w = [0] * X.shape[1]
        m = y.shape[0]
        t = 1

        for _ in range(self.max_iter):
            rand_ind = np.random.choice(range(m), m, replace=False)
            for i in rand_ind:
                w0 = self.w[:-1]
                w0 = np.append(w0, 0)
                if self.which_schedule == 1:
                    step_t = self.step / (1 + (self.step / self.schedule_a) * t)
                elif self.which_schedule == 2:
                    step_t = self.step / (1 + t)
                else:
                    raise ValueError('Please indicate your learning rate schedule')
                yi = y.iloc[i].values
                Xi = X.iloc[i].values
                predicted = yi * np.dot(Xi.T, self.w)
                if predicted <= 1:
                    self.w -= step_t * (w0 - self.C * m * yi * Xi)
                else:
                    self.w -= step_t * w0
            t += 1
        print(f'The weight vector is {self.w}')

    ## Predict outcome
    def predict(self, X):
        def predict1(x):
            score = np.dot(x.T, self.w)
            predicted = 1 if score > 0 else -1
            return predicted

        predicts = np.apply_along_axis(predict1, 1, X)
        return predicts
