from log_regression import LogisticModel, predict_probabilities
import pandas as pd
import matplotlib.pyplot as plt

epochs = 1, 5, 10, 50, 100, 500,
dataset = pd.read_csv(f'data/after preprocessing/breast_cancer.csv')
target_name = 'Maligant'
X = dataset.drop(columns=target_name)
y = dataset[[target_name]]

assessment = pd.DataFrame([], columns=['Log-likelihood for GD', 'Log-likelihood for SGD', 'Log-likelihood for IRLS'])
for n_epochs in epochs:
    GD = LogisticModel(X, y)
    SGD = LogisticModel(X, y)
    IRLS = LogisticModel(X, y)
    GD.GD(n_epochs)
    SGD.SGD(n_epochs)
    IRLS.IRLS(n_epochs)
    assessment = assessment.append({'Log-likelihood for GD': GD.log_likelihood(),
                                    'Log-likelihood for SGD': SGD.log_likelihood(),
                                    'Log-likelihood for IRLS': IRLS.log_likelihood()
                                    }, ignore_index=True)
for label in assessment.columns:
    plt.plot(epochs, assessment[label], label=label)
plt.ylim((assessment.min().min(), 0))
plt.legend()
