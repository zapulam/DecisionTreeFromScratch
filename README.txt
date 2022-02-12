### Decision Trees
### *Written by Zachary Pulliam*

___________________________________________________________________________________________________________________________________________________________________________________

This code was written in order to implement a decision tree classifier that will be used to classify 
four synthetic datasets and a dataset of different legendary and non-legendary Pokemon.

The DecisionTree class can be used on any other dataset that is descretized as long as the class label 
is placed in the last column. This can be done in three steps...

1. Simply create a dataset class similar to those in datasets.py, and initialize the dataset class instance.
   - the data is discretized each time it is loaded in these examples, however, a new csv file can be created from discretized
   data if needed on larger datasets to be more efficient
   - ideally, datasets will contain train and test sets, but here they are the same just to test if the tree works
2. Initialize the DecisionTree, passing in the dataframe subset for training and a desired max tree depth.
3. Call DecisionTree.fit() to fit a decision tree to the training data.
(4.) Call DecisionTree.acc() to calculate the accuracy of the tree, passing in the dataframe subset for testing.

In order to visualize the decision surfaces created for the 2D synthetic datasets, use the ipython notebook visualize.ipynb 
with the same variables.

The synthetic dataset contains two features, A and B, which are continuous and need to be discretized.
The Pokemon dataset contains continuous values for speed, health, etc. and Pokemon type data.

The data for these datasets can be explored in the 'exploration' notebook. 
Bins for discretization can be changed in the bins.py file.
