import scipy
import numpy as np
import pickle
import csv

import warnings
warnings.filterwarnings("ignore", message="Recall is ill-defined")
warnings.filterwarnings("ignore", message="F-score is ill-defined")
warnings.filterwarnings("ignore", message="Precision is ill-defined")
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import semi_supervised
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
from sklearn.tree._tree import TREE_LEAF

# Classifiers
NB = "Naive Bayes"
MLP = "Multi-Layer Perceptron"
SVM = "Support Vector Machine"
DT = "Decision Tree"

# datasets = ["ALLAML", "CLL_SUB_111", "colon", "GLI_85", "GLIOMA", "leukemia", "lung", "lung_discrete",
            # "lymphoma", "nci9", "Prostate_GE", "SMK_CAN_187", "TOX_171"]
datasets = ["ALLAML", "colon", "GLIOMA", "leukemia", "lung", "lung_discrete", "lymphoma", "Prostate_GE", "TOX_171"]

def load_data(dataset):
    """Load the dataset, return a pandas dataset object.
    """
    print("Dataset: {}".format(dataset))
    with open("{}_new_wclass2.pkl".format(dataset), "rb") as dataset_file:
        data = pickle.load(dataset_file)
    data['X'] = np.hstack((data['X'], data['Real'].reshape(-1, 1)))
    return data

def calculate_metrics(data, predicted, result):
    """Calculates the accuracy, precision, recall and F1 score for a given set of predictions.
    """
    accuracy = metrics.accuracy_score(data, predicted)
    precision = metrics.precision_score(data, predicted, average='weighted')
    recall = metrics.recall_score(data, predicted, average='weighted')
    f1 = metrics.f1_score(data, predicted, average='weighted')
    result['Accuracy'] = accuracy
    result['Precision'] = precision
    result['Recall'] = recall
    result['F1-Score'] = f1
    print("Accuracy: {} Precision: {} Recall: {} F1 Score: {}".format(accuracy, precision, recall, f1))

def data_split(data, test_size=0.2):
    """Splits the data into a training and test set.
    """
    real_indices = np.where(data['Real'] == 1)
    fake_indices = np.where(data['Real'] == 0)
    real_x = data['X'][real_indices]
    real_y = data['Y'][real_indices]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        real_x,
        real_y,
        test_size=test_size,
        random_state=None,
        stratify=real_y
    )
    x_train = np.vstack((x_train, data['X'][fake_indices]))
    y_train = np.vstack((y_train, data['Y'][fake_indices]))
    return x_train, x_test, np.ravel(y_train), np.ravel(y_test)

def nb():
    print("=====Naive Bayes=====")
    for dataset in datasets:
        result = dict({"Algorithm":NB, "Dataset":dataset})
        data = load_data(dataset)
        nbclassifier = naive_bayes.GaussianNB()
        x_train, x_test, y_train, y_test = data_split(data)
        y_pred = nbclassifier.fit(x_train, y_train).predict(x_test)
        calculate_metrics(y_pred, y_test, result)
        results.append(result)

def mlp():
    print("=====Multi-Layer Perceptron=====")
    for dataset in datasets:
        result = dict({"Algorithm":MLP, "Dataset":dataset})
        data = load_data(dataset)
        mlpclassifier = neural_network.MLPClassifier(
            solver='lbfgs',  # Recommended by scikit-learn for smaller datasets
            hidden_layer_sizes=(1000,100,10)
        )
        x_train, x_test, y_train, y_test = data_split(data)
        y_pred = mlpclassifier.fit(x_train, y_train).predict(x_test)
        calculate_metrics(y_pred, y_test, result)
        results.append(result)

def support_vector_machine():
    print("=====Support Vector Machine=====")
    for dataset in datasets:
        result = dict({"Algorithm":SVM, "Dataset":dataset})
        data = load_data(dataset)
        svmclassifier = svm.SVC(kernel='rbf', gamma='auto')
        x_train, x_test, y_train, y_test = data_split(data)
        y_pred = svmclassifier.fit(x_train, y_train).predict(x_test)
        calculate_metrics(y_pred, y_test, result)
        results.append(result)

def prune_index(inner_tree, index, threshold):
    """Source: https://stackoverflow.com/a/49496027/5760608
    """
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)

def decision_tree():
    print("=====Decision Tree=====")
    for dataset in datasets:
        result = dict({"Algorithm":DT, "Dataset":dataset})
        data = load_data(dataset)
        treeclassifier = tree.DecisionTreeClassifier(criterion='gini')
        x_train, x_test, y_train, y_test = data_split(data)
        y_pred = treeclassifier.fit(x_train, y_train).predict(x_test)
        prune_index(treeclassifier.tree_, 0, 5)
        calculate_metrics(y_pred, y_test, result)
        results.append(result)

# Get results and save them to file
results = list()

for i in range(10):
    nb()
    mlp()
    support_vector_machine()
    decision_tree()
    print("===== End of iteration: ", i)

print("saving results")
with open('augmented_data_results_wclass2.csv', 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, results[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(results)
