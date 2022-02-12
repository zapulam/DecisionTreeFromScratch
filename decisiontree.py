"""
Written by Zachary Pulliam

Contains the Node and DecisionTree classes which are the heart of the work here. The DecisionTree class, when called,
will create Nodes as the tree grows until the data is exhausted or a desired tree depth is met.
"""

import math
import numpy as np



"""Contains the information of the nodes of the Decision Tree."""
class Node:
    def __init__(self):
        self.head = False  # True for the head node only
        self.data = None  # data at each node, split recursively
        self.feat = None  # feature split on (if node is used to split on)
        self.feats = None  # possible features remaining
        self.value = None  # feature value
        self.label = None  # leaf label (if node is a leaf)
        self.childs = None  # list of children 


        
"""Decision Tree Classifier using ID3 algorithm."""
class DecisionTree:
    def __init__(self, df, max_depth):
        self.df = df  # dataset
        self.features = list(df)[0:-1]  # name of the features
        self.label = list(df)[-1]  # name of label column
        self.values = df[self.label].unique()  # unique values
        self.depth = 0  # total depth of tree
        self.max_depth = max_depth  # max depth desired
        self.node = None  # nodes
        self.entro = self.entropy(df[self.label].values)  # system entropy


    """calculate entropy for given column"""
    def entropy(self, column):
        counts = np.bincount(column) # count unique values in the column
        probs = counts / len(column) # divide by total column length

        entropy = 0
        for prob in probs: # add each column's entropy to the total entropy
            if prob > 0:
                entropy += prob * math.log(prob, 2)
        
        return -entropy


    """calculate information gain for given column and class label"""
    def gain(self, data, split, label):
        entro = self.entropy(data[label]) # total entropy
        vals = data[split].unique() # count unique values in the column
        
        subs = {}
        for i in vals: # subsets of data for each unique val
            subs["sub{0}".format(i)] = data[data[split] == i]

        subtract = 0
        for sub in subs: # loop through subsets and calculate each's entropy
            prob = (subs[sub].shape[0] / data.shape[0]) 
            subtract += prob * self.entropy(subs[sub][label])
        
        return entro - subtract  # return information gain


    """return highest information gain"""
    def max_gain(self, df, cols, label):
        gains = {}
    
        for col in cols:
            col_gain = self.gain(df, col, label)  # find the information gain for column                                         
            gains[col] = col_gain  # add information gain to dict
                                           
        return max(gains, key=gains.get)
    

    """initially calls ID3 algorithm to create nodes for the decision tree"""
    def fit(self):
        self.node = self._id3(self.df, self.features, self.node)  # define head node
        self.clean(self.node)  # post-process nodes


    """ID3 algorithm: called recursively until criteria is met, returns head node of the decision tree"""
    def _id3(self, df, features, node):
        self.depth += 1

        if not node:
            node = Node()  # initialize node
            node.head = True  # first node = head
            node.feats = features  # features left 

        node.data = df  # data for each node

        label_vals = df[self.label].tolist()  # all label values

        if len(set(label_vals)) == 1:  # if label set is pure, return node
            node.label = label_vals[0]  # leaf value 
            return node

        if len(node.feats) == 0 or self.depth > self.max_depth:  # if no more features, or max depth is met, return node
            node.label = max(set(label_vals), key=label_vals.count)  # leaf becomes most common label value
            return node

        best_feature = self.max_gain(df, features, self.label)  # choose feature that maximizes information gain
        node.feat = best_feature  # feature which node will split data on
        node.childs = []  # initialize list of children

        feature_vals = list(set(self.df[best_feature]))  # possible values of best feature
        for val in feature_vals:  # loop through all values
            child = Node()  # initialize child for each value
            child.value = val  # feature value of child node
            node.childs.append(child)  # append new child node to current node's children

            new_df = (df.loc[df[best_feature] == val]).drop(columns=[best_feature])  # split data on best feature
            if new_df.empty:  # if there are no instances
                child.label = max(set(label_vals), key=label_vals.count)  # leaf becomes most common value
            else:  # else, ID3 is called again for new dataframe
                if node.feats and best_feature in node.feats:
                    child.feats = node.feats.copy()
                    child.feats.remove(best_feature)
                child = self._id3(new_df, child.feats, child)  # recursively call ID3
                self.depth -= 1
        return node  # return head node


    """traverses the tree and removes children from a node if they all have the same class label and replaces label to parent node"""
    def clean(self, node):
        if node.childs != None:
            if all(child.label == node.childs[0].label for child in node.childs) and node.childs[0].label != None:  # if all labels are the same and not None (not leaf)
                node.label = node.childs[0].label
                node.childs = []
                node.feat = None
            else:
                for child in node.childs:
                    if child.feat != None:
                        self.clean(child)  # recursively clean all nodes


    """initially calls _acc with the head node for each row in the df, tracks correct decisions"""
    def acc(self, df):
        correct = 0  # total correct
        node = self.node  # head node
        for i in range(len(df.index)):  # for all idxs in df
            correct += self._acc(df.iloc[[i]], node)  # plus 0 or 1 depending
        return correct/len(df.index)


    """Accuracy algorithm: predicts label for each value and returns correct or incorrect"""
    def _acc(self, row, node):
        if node.label == None:  # if node does not have label, go to childs
            val = row.iloc[0][node.feat]
            for child in node.childs:
                if child.value == val:
                    return self._acc(row, child)  # recursively call till leaf is found
        elif node.label == row.iloc[0][self.label]:
            return 1  # correct
        else:
            return 0  # incorrect


    """Print Tree Algorithm: initiates recursive call of _print_tree to print full decision tree"""
    def print_tree(self):
        print('')
        self._print_tree(self.node, count=1)
        print('')


    """recursively prints the decision tree"""
    def _print_tree(self, node, count):
        if node.head == True:
            print(node.feat)  # print head node
        if node.childs != None:
            for child in node.childs:  # recurse through childs
                for i in range(count): print('       ', end='')
                if child.feat != None:
                    print(child.value, '->', child.feat)
                    self._print_tree(child, count+1)
                else:
                    print(child.value, '->', child.label)
