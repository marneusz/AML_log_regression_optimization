from sklearn.model_selection import train_test_split

from log_regression import LogisticModel, accuracy, recall, precision
import pandas as pd
import matplotlib.pyplot as plt

epochs = (
    1,
    3,
    5,
    10,
    50,
    100,
)
dataset = pd.read_csv(f"data/after preprocessing/breast_cancer.csv")
target_name = "Maligant"
X = dataset.drop(columns=target_name)
y = dataset[[target_name]]
X_train, X_test, y_train, y_test = train_test_split(X, y)
loglikelihoods = pd.DataFrame([], columns=["GD", "SGD", "IRLS"])
accuracies = pd.DataFrame([], columns=["GD", "SGD", "IRLS"])
recalls = pd.DataFrame([], columns=["GD", "SGD", "IRLS"])
precisions = pd.DataFrame([], columns=["GD", "SGD", "IRLS"])
for n_epochs in epochs:
    GD = LogisticModel(X_train, y_train)
    SGD = LogisticModel(X_train, y_train)
    IRLS = LogisticModel(X_train, y_train)
    GD.GD(n_epochs)
    SGD.SGD(n_epochs)
    IRLS.IRLS(n_epochs)
    loglikelihoods = loglikelihoods.append(
        {
            "GD": GD.log_likelihood(),
            "SGD": SGD.log_likelihood(),
            "IRLS": IRLS.log_likelihood(),
        },
        ignore_index=True,
    )
    accuracies = accuracies.append(
        {
            "GD": accuracy(GD.fit(X_test), y_test),
            "SGD": accuracy(SGD.fit(X_test), y_test),
            "IRLS": accuracy(IRLS.fit(X_test), y_test),
        },
        ignore_index=True,
    )
    recalls = recalls.append(
        {
            "GD": recall(GD.fit(X_test), y_test),
            "SGD": recall(SGD.fit(X_test), y_test),
            "IRLS": recall(IRLS.fit(X_test), y_test),
        },
        ignore_index=True,
    )
    precisions = precisions.append(
        {
            "GD": precision(GD.fit(X_test), y_test),
            "SGD": precision(SGD.fit(X_test), y_test),
            "IRLS": precision(IRLS.fit(X_test), y_test),
        },
        ignore_index=True,
    )
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for label in loglikelihoods.columns:
    axs[0][0].plot(epochs, loglikelihoods[label], label=label)
axs[0][0].set_ylim((loglikelihoods.min().min(), 0))
axs[0][0].set_title("Loglikelihood")
axs[0][0].legend()

for label in accuracies.columns:
    axs[0][1].plot(epochs, accuracies[label], label=label)
axs[0][1].set_title("Accuracy")
axs[0][1].legend()

for label in recalls.columns:
    axs[1][1].plot(epochs, recalls[label], label=label)
axs[1][1].set_title("Recall")
axs[1][1].legend()

for label in precisions.columns:
    axs[1][0].plot(epochs, precisions[label], label=label)
axs[1][0].set_title("Precision")
axs[1][0].legend()
