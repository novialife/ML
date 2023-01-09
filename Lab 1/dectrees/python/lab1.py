import random

import monkdata as m
import dtree as d
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate


# Calculates the entropy of the given training_set
def entropy(training_set):
    return d.entropy(training_set)


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def get_best_tree(currtree, monkval):
    found_better_tree = False

    for newtree in d.allPruned(currtree):
        if d.check(newtree, monkval) > d.check(currtree, monkval):
            found_better_tree = True
            currtree = newtree

    if found_better_tree:
        currtree = get_best_tree(currtree, monkval)
    return currtree


def main():
    training_sets = [m.monk1, m.monk2, m.monk3]
    test_sets = [m.monk1test, m.monk2test, m.monk3test]

    # Assignment 1
    print("Assignment 1")
    i = 1
    print("Dataset Entropy")
    for training_set in training_sets:
        print("MONK-" + str(i), end="  ")
        print(entropy(training_set))
    print()

    # Assignment 3
    print("Assignment 3")
    print("Dataset", "a1", "a2", "a3", "a4", "a5", "a6")
    for i in range(0, 3):
        training_set = training_sets[i]
        print("MONK-" + str(i + 1), end=" ")
        for j in range(6):
            print(d.averageGain(training_set, m.attributes[j]), end="  ")
        print()
    print()

    # Assignment 5
    print("Assignment 5")
    """ 
    Split the monk1 data into subsets according to the selected attribute using
    the function select (again, defined in d.py) and compute the information gains for the nodes on the next level of the tree. Which attributes
    should be tested for these nodes?

    For the monk1 data draw the decision tree up to the first two levels and
    assign the majority class of the subsets that resulted from the two splits
    to the leaf nodes. You can use the predefined function mostCommon (in
    d.py) to obtain the majority class for a dataset.
    Now compare your results with that of a predefined routine for ID3. Use
    the function buildTree(data, m.attributes) to build the decision tree.
    If you pass a third, optional, parameter to buildTree, you can limit the
    depth of the generated tree.

    You can use print to print the resulting tree in text form, or use the
    function drawTree from the file drawtree_qt4.py or drawtree_qt5.py,
    depending on your PyQt version, to draw a graphical representation.

    """

    # WHAT IS THIS ^^??
    # Code below is 100% wrong
    print("        E_train", " E_test")
    for i in range(0, 3):
        print("MONK-" + str(i + 1), end="  ")
        t = d.buildTree(training_sets[i], m.attributes)
        print(d.check(t, training_sets[i]), end="      ")
        print(d.check(t, test_sets[i]))

    # Assignment 7 Pruning
    dataset_names = ('MONK-1', 'MONK-3')
    datasets = (m.monk1, m.monk3)
    datasets_test = (m.monk1test, m.monk3test)

    fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    decimals = 3
    n = 10000

    header_pruned = ['Fraction', 'Mean error (n = {})'.format(n), 'Standard deviation']
    header = ['Fraction', 'Mean error Non-Pruned (n = {})'.format(n), 'Standard deviation']

    for dataset_name, dataset, dataset_test in zip(dataset_names, datasets, datasets_test):
        data_pruned = []
        mean_errors_pruned = []
        stdev_pruned = []

        data = []
        mean_errors = []
        stdev = []

        for fraction in fractions:

            errors_pruned = []
            errors = []

            for i in range(n):
                monktrain, monkval = partition(dataset, fraction)
                built_tree = d.buildTree(monktrain, m.attributes)
                best_tree = get_best_tree(built_tree, monkval)

                errors_pruned.append(1 - d.check(best_tree, dataset_test))
                errors.append((1 - d.check(built_tree, monkval)))  # Un-prouned value

            mean_error_pruned = round(statistics.mean(errors_pruned), decimals)
            mean_errors_pruned.append(mean_error_pruned)

            mean_error = round(statistics.mean(errors), decimals)
            mean_errors.append(mean_error)

            stdev_pruned.append(round(statistics.stdev(errors_pruned), decimals))
            stdev.append(round(statistics.stdev(errors), decimals))

            data_pruned.append([fraction, mean_error_pruned, statistics.mean(stdev_pruned)])
            data.append([fraction, mean_error, statistics.mean(stdev)])

        print(tabulate(data_pruned, header_pruned), '\n')
        print(tabulate(data, header), '\n')

        plt.errorbar(fractions, mean_errors_pruned, yerr=stdev_pruned, marker='o', label="Pruned Dataset")
        plt.errorbar(fractions, mean_errors, yerr=stdev, marker='o', label="Non-Pruned Dataset")
        plt.legend()

        plt.title('{} (n = {})'.format(dataset_name, n))
        plt.xlabel('Fractions')
        plt.ylabel('Mean errors')
        plt.show()


main()
