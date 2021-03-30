import pandas as pd
import numpy as np


cancer = pd.read_csv('data/breast_cancer.csv')
candy = pd.read_csv('data/candy.csv')
nba = pd.read_csv('data/nba.csv')
wine = pd.read_csv('data/wine.csv')


# this dataset has null values in one column ('EXP_CEO')
# I will create a column indicating if the value is missing and replace NA's with mean
bankrupcy = pd.read_csv('data/bankrupt.csv')
bankrupcy['EXP_CEO_NA'] = bankrupcy['EXP_CEO'].isna().astype(int)
bankrupcy['EXP_CEO'] = bankrupcy['EXP_CEO'].replace(np.nan, bankrupcy['EXP_CEO'].mean())

