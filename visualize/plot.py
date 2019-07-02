__author__ = 'Horace'

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.learning_curve import learning_curve, validation_curve


BAR_PATH = "./graph/bar_chart_"
HIST_PATH = "./graph/histogram_"
SCATTER_PATH = "./graph/scatter_"

def class_to_number(target_array):
    number_array = []
    target_list = list(set(target_array))
    for element in target_array:
        number_array.append(target_list.index(element))

    return target_list, np.array(number_array)


def plot_pdf(figure, header_name, filename):
    pp = PdfPages(header_name + filename + ".pdf")
    pp.savefig(figure)
    pp.close()

# binary, nominal, class variable
def bar(dataset, index, type_name="", rotation=0):
    width = 0.5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    index_name = dataset.columns[index-1]
    x = dataset[dataset.columns[index-1]].value_counts()
    n = np.arange(x.shape[0])  # location of bar
    ax.bar(left=n, height=list(x), width=width)
    plt.title('Distribution of ' + index_name + " (" + type_name + ")")
    plt.xticks(n + width/2., tuple(x.keys()), rotation=rotation)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.15, box.width, box.height * 0.8])
    file_name = index_name + "_" + type_name
    plot_pdf(figure=fig, header_name=BAR_PATH, filename=file_name)
    plt.show()


def hist(dataset, index, bins, min_threshold=None, max_threshold=None, type_name=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    index_name = dataset.columns[index-1]
    x = dataset[dataset.columns[index-1]]
    x_clean = [value for value in x if type(value).__name__ != "str" and pd.isnull(value) == False]
    if min_threshold is None:
        min_threshold = min(x_clean)

    if max_threshold is None:
        max_threshold = max(x_clean)

    x_selected = [value for value in x_clean if value <= max_threshold and value >= min_threshold]
    ax.hist(x_selected, bins=bins)
    plt.title('Distribution of ' + index_name + " (" + type_name + ")")
    file_name = index_name + "_" + type_name
    plot_pdf(figure=fig, header_name=HIST_PATH, filename=file_name)
    plt.show()


def scatter(dataset, x_index, y_index, color_index=None, x_class_list=None, type_name=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # data_array = np.array(dataset)
    x_index_name = dataset.columns[x_index-1]
    y_index_name = dataset.columns[y_index-1]
    if color_index is not None:
        # target_list, color_list = class_to_number(data_array[:,color_index-1])
        # target_list = list(set(data_array[:,color_index-1]))
        target_list = list(set(dataset[dataset.columns[color_index-1]]))
        print target_list
        color_list = mcm.rainbow(np.linspace(0,1,len(target_list)))
        for i in range(len(target_list)):
            classified_data = dataset[dataset[dataset.columns[color_index-1]] == target_list[i]]
            x_list = list(classified_data[classified_data.columns[x_index-1]])
            y_list = list(classified_data[classified_data.columns[y_index-1]])
            ax.scatter(x_list, y_list, c=color_list[i], label=target_list[i])

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    else:
        ax.scatter(list(dataset[dataset.columns(x_index-1)]), list(dataset[dataset.columns(y_index-1)]))

    if x_class_list is None:
        plt.xlabel(x_index_name)
    else:
        plt.xlabel(x_index_name + " " + str(x_class_list))
    plt.ylabel(y_index_name)
    plt.title('Scatter plot of ' + y_index_name + ' against ' + x_index_name + " (" + type_name + ")")
    file_name = y_index_name + "_against_" + x_index_name + "_" + type_name
    plot_pdf(figure=fig, header_name=SCATTER_PATH, filename=file_name)

    plt.show()

def plot_learning_curve(classifier, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Learning Curve")
    plt.ylim((0,1))
    plt.xlabel("Training Example")
    plt.ylabel("Score")
    train_sizes, train_scores, validation_scores = learning_curve(classifier, X, y)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="g")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color="r")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="g", label="Training Score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="r", label="Cross-validation Score")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()

def plot_validation_curve(classifier, X, y, param_name="gamma", param_range=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Validation Curve")
    plt.ylim((0,1))
    plt.xlabel("degree")
    plt.ylabel("Score")

    param_range = np.logspace(-6, 0, 5)
    train_scores, validation_scores = validation_curve(classifier, X, y, param_name, param_range=param_range)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.semilogx(param_range, train_scores_mean, label="Training Score", color="g")
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="g")

    plt.semilogx(param_range, validation_scores_mean, label="Cross-validation Score", color="r")
    plt.fill_between(param_range, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std,
                 alpha=0.2, color="r")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()

    plt.show()






