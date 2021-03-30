import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer = pd.read_csv('data/breast_cancer.csv')
candy = pd.read_csv('data/candy.csv')
nba = pd.read_csv('data/nba.csv')
wine = pd.read_csv('data/wine.csv')


# this dataset has null values in one column ('EXP_CEO')
# I will create a column indicating if the value is missing and replace NA's with mean
bankrupcy = pd.read_csv('data/bankrupt.csv')
bankrupcy['EXP_CEO_NA'] = bankrupcy['EXP_CEO'].isna().astype(int)
bankrupcy['EXP_CEO'] = bankrupcy['EXP_CEO'].replace(np.nan, bankrupcy['EXP_CEO'].mean())

sns.heatmap(cancer.corr(), annot=True, fmt='.2f')
plt.show()
sns.heatmap(candy.corr(), annot=True, fmt='.2f')
plt.show()
sns.heatmap(nba.corr(), annot=True, fmt='.2f')
plt.show()
sns.heatmap(wine.corr(), annot=True, fmt='.2f')
plt.show()
sns.heatmap(bankrupcy.corr(), annot=True, fmt='.2f')
plt.show()