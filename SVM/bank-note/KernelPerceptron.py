import pandas as pd
import numpy as np
from scipy import optimize

def load_csv(filepath):
    features = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
    outcome = ['Outcome']
    dataset = pd.read_csv(filepath, names=features+outcome)
    x = dataset[features]
    y = dataset[outcome]
    y.loc[y['Outcome'] == 0, 'Outcome'] = -1
    x = x.to_numpy()
    y = y['Outcome'].to_numpy()
    return x, y

def gaussian(x1, x2, gamma):
    diff = x1 - x2
    return np.exp(-np.dot(diff, diff) / gamma)

def linear(x1, x2, gamma):
    return np.dot(x1, x2)

class KernelSVM:

    def __init__(self, C, gamma, kernel, max_iter):
        self.alpha = None
        self.w = None
        self.b = None
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.kernel = kernel
        self.supportVectors = None
        self.supportY = None
        self.supportAlphaY = None

    def fit(self, X, y):
        print('Currently using kernel perceptron...')
        N = len(y)
        hXX = np.apply_along_axis(lambda x1: np.apply_along_axis(lambda x2: self.kernel(x1, x2, self.gamma), 1, X), 1, X)
        yp = y.reshape(-1, 1)
        GramHXy = hXX * np.matmul(yp, yp.T)

        # Lagrange dual problem
        def lagr(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))

        # Partial derivate of Ld on alpha
        def lagr_alpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))
        constraints = ({'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                       {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

        # Maximize by minimizing the opposite
        optRes = optimize.minimize(fun=lambda a: -lagr(GramHXy, a),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda a: -lagr_alpha(GramHXy, a),
                                   constraints=constraints,
                                   options={'maxiter': self.max_iter})
        self.alpha = optRes.x
        epsilon = 1e-6
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportY = y[supportIndices]
        self.supportAlphaY = self.supportY * self.alpha[supportIndices]
        if self.kernel == linear:
            self.w = np.sum((self.alpha[:, np.newaxis] * X * y[:, np.newaxis]), axis=0)
            self.b = np.mean([self.supportY[i] - np.matmul(self.supportVectors[i].T, self.w) for i in range(len(self.supportY))])
            print(f'The weight vector is {self.w}')
            print(f'The bias term is {self.b}')

    def predict(self, X):
        def predict1(x):
            x1 = np.apply_along_axis(lambda s: self.kernel(s, x, self.gamma), 1, self.supportVectors)
            x2 = x1 * self.supportAlphaY
            return np.sum(x2)

        d = np.apply_along_axis(predict1, 1, X)
        return 2 * (d > 0) - 1
