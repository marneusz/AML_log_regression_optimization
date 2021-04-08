from sklearn.model_selection import train_test_split

from log_regression import LogisticModel
import pandas as pd
import matplotlib.pyplot as plt

epochs = 1, 5, 10, 50, 100, 500,
dataset = pd.read_csv(f'data/after preprocessing/breast_cancer.csv')
target_name = 'Maligant'
X = dataset.drop(columns=target_name)
y = dataset[[target_name]]
X_train, X_test, y_train, y_test = train_test_split(X, y)
loglikelihoods = pd.DataFrame([], columns=['Log-likelihood for GD', 'Log-likelihood for SGD', 'Log-likelihood for IRLS'])
for n_epochs in epochs:
    GD = LogisticModel(X_train, y_train)
    SGD = LogisticModel(X_train, y_train)
    IRLS = LogisticModel(X_train, y_train)
    GD.GD(n_epochs)
    SGD.SGD(n_epochs)
    IRLS.IRLS(n_epochs)
    loglikelihoods = loglikelihoods.append({'Log-likelihood for GD': GD.log_likelihood(),
                                    'Log-likelihood for SGD': SGD.log_likelihood(),
                                    'Log-likelihood for IRLS': IRLS.log_likelihood()
                                            }, ignore_index=True)

for label in loglikelihoods.columns:
    plt.plot(epochs, loglikelihoods[label], label=label)
plt.ylim((loglikelihoods.min().min(), 0))
plt.legend()
