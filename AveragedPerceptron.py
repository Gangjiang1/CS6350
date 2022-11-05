import pandas as pd
import numpy as np

## load csv files
def load_csv(filepath):
    features = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    outcome = ['Outcome']
    dataset = pd.read_csv(filepath, names=features+outcome)
    x = dataset[features]
    y = dataset[outcome]
    x['Intercept'] = 1
    y.loc[y['Outcome'] == 0, 'Outcome'] = -1
    return x, y

## Build averaged perceptron algorithm
def learn(x, y, alpha, max_iter):
    print('Currently using Averaged Perceptron...')
    np.random.seed(123)
    w = [0] * x.shape[1]
    m = y.shape[0]
    a = 0
    for _ in range(max_iter):
        rand_ind = np.random.choice(range(m), m, replace=False)
        for i in rand_ind:
            yi = y.iloc[i].values
            xi = x.iloc[i].values
            predicted = yi * np.dot(xi.T, w)
            if predicted <= 0:
                w += alpha * yi * xi
            a += w
    print(f'The weight vector is {a}')
    return a

## Predict outcome
def predict(x, w):
    score = np.dot(x.T, w)
    predicted = 1 if score > 0 else -1
    return predicted
