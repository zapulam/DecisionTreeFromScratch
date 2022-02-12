"""
Written by Zachary Pulliam

Conatins two functions, 'run,' which creates a DecisionTree for a synthetic dataset and the pokemon dataset, 
    and 'run_folds,' which will n DecisionTree's for n folds on a synthetic dataset.
"""

from decisiontree import DecisionTree
from datasets import SyntheticDataset, FoldedSyntheticDataset, PokemonDataset


# variables
ROOT = 'C:/Users/Zack\'s PC/Documents/UK Courses/CS 460/Assignment 1/data'  # Root to data files, both synthetic and pokemon data
depth = 3  # max depth of trees
k = 5 # number of bins to create for each variable, must be greater than 0

"""synthetic dataset variables"""
frame = 4  # 1-4, selects which synthetic data file to use

"""synthetic dataset variables for fold-based validation"""
folds = 10  # number of folds


"""main program, creates decision tree for Synthetic Dataset and Pokemon Dataset and reports accuracy of each"""
def run(ROOT, folds):
    # creating datasets
    data = SyntheticDataset(ROOT, frame, k)  # Synthetic Dataset class instance
    folded_data = FoldedSyntheticDataset(ROOT, frame, folds, k)  # Full Synthetic Dataset class instance
    poke_data = PokemonDataset(ROOT, k)  # Pokemon Dataset class instance

    #dfs = [syn_data, fsyn_data, poke_data]
    dfs = [data, poke_data]
    print('')

    for df in dfs:
        # creating and fitting tree for Synthetic Dataset
        tree = DecisionTree(df.train, depth)  # creates instance of DecisonTree
        tree.fit()  # train DecisionTree
        print(df.name, "----------------------------------------------------------------")
        tree.print_tree()  # print tree
        print("Accuracy =", tree.acc(df.test), "\n\n")


"""main, program for n number of folds on a synthtic dataset"""
def run_folds(ROOT, frame, folds, k):
    folded_data = FoldedSyntheticDataset(ROOT, frame, folds, k)  # Full Synthetic Dataset class instance

    print(folded_data.name, "----------------------------------------------------------------")
    for i in range(folds):
        tree = DecisionTree(folded_data.trains[i], depth)  # creates instance of DecisonTree
        tree.fit()  # train DecisionTree
        #tree.print_tree()  # print tree
        print("Accuracy {x} =".format(x=i), tree.acc(folded_data.tests[i]), "\n")


if __name__ == "__main__":
    run(ROOT, folds)
    run_folds(ROOT, frame, folds, k)