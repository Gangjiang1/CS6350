import matplotlib.pyplot as plt

def plot_classifier(niters, train_errors, test_errors):
    plt.plot(niters, train_errors, label='Training Error')
    plt.plot(niters, test_errors, label='Test Error')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.show()

def plot_regressor(niters, costs):
    plt.plot(niters, costs)
    plt.xlabel('Number of Updates')
    plt.ylabel('Cost')
    plt.title('Cost Function of Training by Iteration')
    plt.show()
