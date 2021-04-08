import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

cancer = pd.read_csv("data/before preprocessing/breast_cancer.csv")
candy = pd.read_csv("data/before preprocessing/candy.csv")
nba = pd.read_csv("data/before preprocessing/nba.csv")
wine = pd.read_csv("data/before preprocessing/wine.csv")
bankrupcy = pd.read_csv("data/before preprocessing/bankrupcy.csv")

# filling all missing values with mean value (column-wise)
cancer = cancer.fillna(cancer.mean())
candy = candy.fillna(candy.mean())
nba = nba.fillna(nba.mean())
wine = wine.fillna(wine.mean())
bankrupcy = bankrupcy.fillna(bankrupcy.mean())

# One hot  encoding
encoder = preprocessing.OneHotEncoder()
sector = encoder.fit_transform(bankrupcy[["SECTOR"]])
bankrupcy = pd.concat(
    [bankrupcy, pd.DataFrame(sector.todense(), columns=encoder.categories_[0])], axis=1
)
bankrupcy = bankrupcy.drop(columns="SECTOR")
candy = candy.drop(columns="competitorname")
# dropping string column
nba = nba.drop(columns="Name")

# Corplots before decorrelation
plt.figure(figsize=(10, 8))
sns.heatmap(cancer.corr(), annot=True, fmt=".2f")
plt.title("The cancer set before decorrelation")
# Uniformity of cell size is highly correlated with others
plt.figure(figsize=(10, 8))
sns.heatmap(candy.corr(), annot=True, fmt=".2f")
plt.title("The candy set before decorrelation")
plt.figure(figsize=(10, 8))
sns.heatmap(nba.corr(), annot=True, fmt=".2f")
plt.title("The nba set before decorrelation")
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), annot=True, fmt=".2f")
plt.title("The wine set before decorrelation")
plt.figure(figsize=(10, 8))
sns.heatmap(bankrupcy.corr(), annot=True, fmt=".2f")
plt.title("The bankrupcy set before decorrelation")
plt.show()

cancer = cancer.drop(columns=["Uniformity of Cell Size"])
candy = candy.drop(columns=["fruity"])
nba = nba.drop(columns=["MIN", "PTS", "FGM", "FTA", "DREB", "REB", "3PA"])
wine = wine.drop(columns=["Flavanoids"])
bankrupcy = bankrupcy.drop(columns=["MARKETING_SPENDING"])


# saving

candy.to_csv("data/after preprocessing/candy.csv", index=False)
cancer.to_csv("data/after preprocessing/breast_cancer.csv", index=False)
nba.to_csv("data/after preprocessing/nba.csv", index=False)
wine.to_csv("data/after preprocessing/wine.csv", index=False)
bankrupcy.to_csv("data/after preprocessing/bankrupcy.csv", index=False)

# Corplots after decorelation
plt.figure(figsize=(10, 8))
sns.heatmap(cancer.corr(), annot=True, fmt=".2f")
plt.title("The cancer set after decorrelation")
# Uniformity of cell size is highly correlated with others
plt.figure(figsize=(10, 8))
sns.heatmap(candy.corr(), annot=True, fmt=".2f")
plt.title("The candy set after decorrelation")
plt.figure(figsize=(10, 8))
sns.heatmap(nba.corr(), annot=True, fmt=".2f")
plt.title("The nba set after decorrelation")
plt.figure(figsize=(10, 8))
sns.heatmap(wine.corr(), annot=True, fmt=".2f")
plt.title("The wine set after decorrelation")
plt.figure(figsize=(10, 8))
sns.heatmap(bankrupcy.corr(), annot=True, fmt=".2f")
plt.title("The bankrupcy set after decorrelation")
plt.show()
