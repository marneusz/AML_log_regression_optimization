import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from log_regression import LogisticModel, accuracy, recall, precision, F_measure
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def optimal_KNN(X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_test: pd.DataFrame,
                metric=accuracy,
                max_neighbors: int = 50,
                step: int = 1,
                verbose = False
                ):
    measures = []
    for i in range(1, max_neighbors, step):
        KNN_model = KNeighborsClassifier(n_neighbors=i).fit(X_train, np.ravel(y_train))
        y_pred = KNN_model.predict(X_test).reshape(-1,1)
        measures.append(float(metric(y_test, y_pred)))
    best_K = measures.index(max(measures))
    if verbose:
        print("Maximum metric: ", max(measures), " at K = ", best_K)
    return best_K


def plot_algorithms_comparison(
    dataset=pd.read_csv(f"data/after preprocessing/breast_cancer.csv"),
    target_name="Maligant",
    print_progress=False,
    eps=1e-8,
    figsize=(10, 6)
):
    X = dataset.drop(columns=target_name)
    y = dataset[[target_name]]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    algorithms = ["GD", "SGD", "IRLS", "LDA", "QDA", "KNN"]
    times = []

    start_time = time.time()
    GD = LogisticModel(X_train, y_train)
    GD.GD(1000, eps=eps)
    GD_pred = GD.fit(X_test)
    times.append(time.time() - start_time)
    if print_progress:
        print("GD Done!")

    start_time = time.time()
    SGD = LogisticModel(X_train, y_train)
    SGD.SGD(1000, eps=eps)
    SGD_pred = SGD.fit(X_test)
    times.append(time.time() - start_time)
    if print_progress:
        print("SGD Done!")

    start_time = time.time()
    IRLS = LogisticModel(X_train, y_train)
    IRLS.IRLS(1000, eps=eps)
    IRLS_pred = IRLS.fit(X_test)
    times.append(time.time() - start_time)
    if print_progress:
        print("IRLS Done!")

    start_time = time.time()
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, np.ravel(y_train))
    LDA_pred = LDA.predict(X_test).reshape(-1, 1)
    times.append(time.time() - start_time)
    if print_progress:
        print("LDA Done!")

    start_time = time.time()
    QDA = QuadraticDiscriminantAnalysis()
    QDA.fit(X_train, np.ravel(y_train))
    QDA_pred = QDA.predict(X_test).reshape(-1, 1)
    times.append(time.time() - start_time)
    if print_progress:
        print("QDA Done!")

    start_time = time.time()
    KNN = KNeighborsClassifier(n_neighbors=optimal_KNN(X_train,
                                                       y_train,
                                                       X_test,
                                                       y_test,
                                                       )
                               )
    KNN.fit(X_train, np.ravel(y_train))
    KNN_pred = KNN.predict(X_test).reshape(-1, 1)
    times.append(time.time() - start_time)
    if print_progress:
        print("KNN Done!")

    accuracies = list(
        (
            accuracy(y_test, GD_pred),
            accuracy(y_test, SGD_pred),
            accuracy(y_test, IRLS_pred),
            accuracy(y_test, LDA_pred),
            accuracy(y_test, QDA_pred),
            accuracy(y_test, KNN_pred),
        )
    )
    accuracies = list(map(float, accuracies))

    recalls = list(
        (
            recall(y_test, GD_pred),
            recall(y_test, SGD_pred),
            recall(y_test, IRLS_pred),
            recall(y_test, LDA_pred),
            recall(y_test, QDA_pred),
            recall(y_test, KNN_pred),
        )
    )
    recalls = list(map(float, recalls))

    precisions = list(
        (
            precision(y_test, GD_pred),
            precision(y_test, SGD_pred),
            precision(y_test, IRLS_pred),
            precision(y_test, LDA_pred),
            precision(y_test, QDA_pred),
            precision(y_test, KNN_pred),
        )
    )
    precisions = list(map(float, precisions))

    F_measures = list(
        (
            F_measure(y_test, GD_pred),
            F_measure(y_test, SGD_pred),
            F_measure(y_test, IRLS_pred),
            F_measure(y_test, LDA_pred),
            F_measure(y_test, QDA_pred),
            F_measure(y_test, KNN_pred),
        )
    )
    F_measures = list(map(float, F_measures))

    R2_measures = list(
        (
            GD.R2_measure(),
            SGD.R2_measure(),
            IRLS.R2_measure(),
            # R2_measure(np.concatenate((LDA.intercept_, np.ravel(LDA.coef_))), X_test, y_test)
        )
    )
    R2_measures = list(map(float, R2_measures))

    fig, axs = plt.subplots(3, 2, figsize=figsize)

    axs[0][0].barh(list(reversed(algorithms)),
                   list(reversed(accuracies)),
                   color=("navy", "blue", "dodgerblue", "firebrick", "red", "lightsalmon"))
    axs[0][0].set_title("Accuracy")

    axs[0][1].barh(list(reversed(algorithms)),
                   list(reversed(recalls)),
                   color=("navy", "blue", "dodgerblue", "firebrick", "red", "lightsalmon"))
    axs[0][1].set_title("Recall")

    axs[1][0].barh(list(reversed(algorithms)),
                   list(reversed(precisions)),
                   color=("navy", "blue", "dodgerblue", "firebrick", "red", "lightsalmon"))
    axs[1][0].set_title("Precision")

    axs[1][1].barh(list(reversed(algorithms)),
                   list(reversed(F_measures)),
                   color=("navy", "blue", "dodgerblue", "firebrick", "red", "lightsalmon"))
    axs[1][1].set_title("F-measure")

    axs[2][0].barh(list(reversed(algorithms))[3:],
                   list(reversed(R2_measures)),
                   color=("navy", "blue", "dodgerblue", "firebrick"))
    axs[2][0].set_title("R2-measure")
    axs[2][0].set_xlim(max(0.0, round(min(R2_measures), 1)-0.1), 1)

    axs[2][1].barh(list(reversed(algorithms)),
                   list(reversed(times)),
                   color=("navy", "blue", "dodgerblue", "firebrick", "red", "lightsalmon"))
    axs[2][1].set_title("Times of Execution")
