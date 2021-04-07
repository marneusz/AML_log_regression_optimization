from log_regression import LogisticModel, predict_probabilities
import pandas as pd
import matplotlib.pyplot as plt

epochs = 10, 50, 100, 500, 1000,
cancer = pd.read_csv(f'data/after preprocessing/breast_cancer.csv')
assessment = pd.DataFrame([], columns=['Log-likelihood for GD', 'Log-likelihood for SGD', 'Log-likelihood for IRLS'])
for n_epochs in epochs:
    GD = LogisticModel(cancer.drop(columns='Maligant'), cancer[['Maligant']])
    SGD = LogisticModel(cancer.drop(columns='Maligant'), cancer[['Maligant']])
    IRLS = LogisticModel(cancer.drop(columns = 'Maligant'), cancer[['Maligant']])
    GD.GD(n_epochs)
    SGD.SGD(n_epochs)
    IRLS.IRLS(n_epochs, print_progress=True)
    assessment = assessment.append({'Log-likelihood for GD': GD.log_likelihood(),
                                    'Log-likelihood for SGD': SGD.log_likelihood(),
                                    'Log-likelihood for IRLS': IRLS.log_likelihood()
                                    }, ignore_index=True)
for label in assessment.columns:
    plt.plot(epochs, assessment[label], label=label)
plt.legend()
