from log_regression import LogisticModel
import pandas as pd
import matplotlib.pyplot as plt


epochs = 10, 50, #100, 500, 1000,
nba = pd.read_csv(f'data/before decorrelation/nba.csv').drop(columns = 'Name')
assessment = pd.DataFrame([], columns=['LL for GD', 'LL for SGD', 'LL for IRLS'])
for n_epochs in epochs:
    GD = LogisticModel(nba.drop(columns = 'TARGET'), nba[['TARGET']])
    SGD = LogisticModel(nba.drop(columns = 'TARGET'), nba[['TARGET']])
    IRLS = LogisticModel(nba.drop(columns = 'TARGET'), nba[['TARGET']])
    GD.GD(n_epochs)
    SGD.SGD(n_epochs)
    IRLS.IRLS(n_epochs, 0.1)
    assessment = assessment.append({'LL for GD': GD.log_likelihood(),
                       'LL for SGD': SGD.log_likelihood() ,
                       'LL for IRLS': IRLS.log_likelihood()}, ignore_index=True)

plt.plot(epochs, assessment)