from log_regression import LogisticModel, predict_probabilities
import pandas as pd
import matplotlib.pyplot as plt


epochs = 10, 50, 100, 500, 1000,
cancer = pd.read_csv(f'data/after preprocessing/breast_cancer.csv').drop(columns = ['Sample code number', 'Bare Nuclei'])
assessment = pd.DataFrame([], columns=['LL for GD', 'LL for SGD'])
for n_epochs in epochs:
    GD = LogisticModel(cancer.drop(columns ='Maligant'), cancer[['Maligant']])
    SGD = LogisticModel(cancer.drop(columns ='Maligant'), cancer[['Maligant']])
    #IRLS = LogisticModel(cancer.drop(columns = 'Target'), cancer[['Target']])
    GD.GD(n_epochs)
    SGD.SGD(n_epochs)
    #IRLS.IRLS(n_epochs, 0.1)
    assessment = assessment.append({'LL for GD': GD.log_likelihood(),
                       'LL for SGD': SGD.log_likelihood() }, ignore_index=True)

plt.plot(epochs, assessment)