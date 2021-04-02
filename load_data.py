import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer = pd.read_csv('data/before decorrelation/breast_cancer.csv')
candy = pd.read_csv('data/before decorrelation/candy.csv')
nba = pd.read_csv('data/before decorrelation/nba.csv')
wine = pd.read_csv('data/before decorrelation/wine.csv')


# this dataset has null values in one column ('EXP_CEO')
# I will create a column indicating if the value is missing and replace NA's with mean
bankrupcy = pd.read_csv('data/before decorrelation/bankrupcy.csv')
bankrupcy['EXP_CEO_NA'] = bankrupcy['EXP_CEO'].isna().astype(int)
bankrupcy['EXP_CEO'] = bankrupcy['EXP_CEO'].replace(np.nan, bankrupcy['EXP_CEO'].mean())

# Corplots before decorelation
plt.figure(figsize=(10, 8))
sns.heatmap(cancer.corr(), annot=True, fmt='.2f')
plt.title('The cancer set before decorrelation')
# Uniformity of cell size is highly correlated with others
plt.figure(figsize=(10, 8))
sns.heatmap(candy.corr(), annot=True, fmt='.2f')
plt.title('The candy set before decorrelation')
plt.figure(figsize=(10, 8))
sns.heatmap(nba.corr(), annot=True, fmt='.2f')
plt.title('The nba set before decorrelation')
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), annot=True, fmt='.2f')
plt.title('The wine set before decorrelation')
plt.figure(figsize=(10, 8))
sns.heatmap(bankrupcy.corr(), annot=True, fmt='.2f')
plt.title('The bankrupcy set before decorrelation')
plt.show()

cancer = cancer.drop(columns=['Uniformity of Cell Size'])
candy = candy.drop(columns=['fruity'])
nba = nba.drop(columns=['MIN', 'PTS', 'FGM', 'FTA', 'DREB', 'REB', '3PA'])
wine = wine.drop(columns=['Flavanoids'])
bankrupcy = bankrupcy.drop(columns=['MARKETING_SPENDING'])


# Corplots after decorelation
plt.figure(figsize=(10, 8))
sns.heatmap(cancer.corr(), annot=True, fmt='.2f')
plt.title('The cancer set after decorrelation')
# Uniformity of cell size is highly correlated with others
plt.figure(figsize=(10, 8))
sns.heatmap(candy.corr(), annot=True, fmt='.2f')
plt.title('The candy set after decorrelation')
plt.figure(figsize=(10, 8))
sns.heatmap(nba.corr(), annot=True, fmt='.2f')
plt.title('The nba set after decorrelation')
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), annot=True, fmt='.2f')
plt.title('The wine set after decorrelation')
plt.figure(figsize=(10, 8))
sns.heatmap(bankrupcy.corr(), annot=True, fmt='.2f')
plt.title('The bankrupcy set after decorrelation')
plt.show()
