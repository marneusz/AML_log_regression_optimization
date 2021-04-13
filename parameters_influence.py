from sklearn.model_selection import train_test_split

from log_regression import LogisticModel, accuracy
import pandas as pd
import matplotlib.pyplot as plt


def plot_GD_SGD(
    learning_rates=(
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.75,
        1
    ),
    batch_sizes=(
            1,
            2,
            4,
            8,
            10,
            16,
            25,
            32,
            50,
            64,
            128,
        ),
    dataset=pd.read_csv(f"data/after preprocessing/breast_cancer.csv"),
    target_name="Maligant",
    figsize=(6, 6)
):
    X = dataset.drop(columns=target_name)
    y = dataset[[target_name]]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    loglikelihoods_lr = pd.DataFrame([], columns=["GD", "SGD"])
    accuracies_lr = pd.DataFrame([], columns=["GD", "SGD"])
    for learning_rate in learning_rates:
        GD = LogisticModel(X_train, y_train)
        SGD = LogisticModel(X_train, y_train)
        GD.GD(n_epochs=5, learning_rate=learning_rate, eps=None)
        SGD.SGD(n_epochs=5, learning_rate=learning_rate, batch_size=16, random_state=None, eps=None)

        loglikelihoods_lr = loglikelihoods_lr.append(
            {
                "GD": GD.log_likelihood(),
                "SGD": SGD.log_likelihood(),
            },
            ignore_index=True,
        )
        accuracies_lr = accuracies_lr.append(
            {
                "GD": accuracy(GD.fit(X_test), y_test),
                "SGD": accuracy(SGD.fit(X_test), y_test),
            },
            ignore_index=True,
        )

    loglikelihoods_bs = pd.DataFrame([], columns=["SGD"])
    accuracies_bs = pd.DataFrame([], columns=["SGD"])
    for batch_size in batch_sizes:
        if batch_size > X.shape[0]:
            print("Too high batch size!")
            break
        SGD = LogisticModel(X_train, y_train)
        SGD.SGD(n_epochs=5, learning_rate=0.1, batch_size=batch_size, random_state=None, eps=None)

        loglikelihoods_bs = loglikelihoods_bs.append(
            {
                "SGD": SGD.log_likelihood(),
            },
            ignore_index=True,
        )
        accuracies_bs = accuracies_bs.append(
            {
                "SGD": accuracy(SGD.fit(X_test), y_test),
            },
            ignore_index=True,
        )

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for label in loglikelihoods_lr.columns:
        axs[0][0].plot(learning_rates, loglikelihoods_lr[label], label=label)
    axs[0][0].set_ylim((loglikelihoods_lr.min().min(), 0))
    axs[0][0].set_title("Loglikelihood vs Learning Rate")
    axs[0][0].legend()
    for label in accuracies_lr.columns:
        axs[0][1].plot(learning_rates, accuracies_lr[label], label=label)
    axs[0][1].set_title("Accuracy vs Learning Rate")
    axs[0][1].legend()
    for label in loglikelihoods_bs.columns:
        axs[1][1].plot(batch_sizes, loglikelihoods_bs[label], label=label)
    axs[1][1].set_title("Loglikelihood vs Batch Size (SGD)")
    axs[1][1].legend()
    for label in accuracies_bs.columns:
        axs[1][0].plot(batch_sizes, accuracies_bs[label], label=label)
    axs[1][0].set_title("Accuracy vs Batch Size (SGD)")
    axs[1][0].legend()

    plt.show()


if __name__ == '__main__':
    plot_GD_SGD()
