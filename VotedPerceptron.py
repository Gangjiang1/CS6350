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

## Build voted perceptron algorithm
def learn(x, y, alpha, max_iter):
    print('Currently using Voted Perceptron...')
    np.random.seed(123)
    w = [0] * x.shape[1]
    m = y.shape[0]
    k = 0
    w_list = [[w, 1]]
    for _ in range(max_iter):
        rand_ind = np.random.choice(range(m), m, replace=False)
        for i in rand_ind:
            yi = y.iloc[i].values
            xi = x.iloc[i].values
            predicted = yi * np.dot(xi.T, w)
            if predicted <= 0:
                w = w + alpha * yi * xi
                k += 1
                w_list.append([w, 1])
            else:
                w_list[k][1] += 1
    print(f'The weight vectors and their counts are {w_list}')
    return w_list

## Predict outcome
def predict(x, w_list):
    scores = []
    for i in range(len(w_list)):
        scorei = np.dot(x.T, w_list[i][0])
        predictedi = 1 if scorei > 0 else -1
        weighted_scorei = w_list[i][1] * predictedi
        scores.append(weighted_scorei)
    score = sum(scores)
    predicted = 1 if score > 0 else -1
    return predicted
