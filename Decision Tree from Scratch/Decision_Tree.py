import pandas as pd
from Leaf import Branches
import numpy as np
from collections import Counter

class Tree(object):
    '''
    Our decision tree. We will call multiple branches on all of our features
    until we have a model that accurately predicts whether a given passenger
    died or survived.
    '''

    def __init__(self):
        self.impurity_criterion = self.entropy #I'll be using entropy as my measure of impurity.
                                                    #alternatively use _gini
        self.root = None
        self.feature_names = None #string names of each X feature

    def fit(self,X,y,feature_names=None):
        '''
        This is where we build the tree to fit the training data. Input a
        list of feature name strings if you want to keep them labeled properly.

        INPUT:
            -X: numpy array of our X data
            -y: 1d numpy array of our training data results (Survived or Died)
            -feature_names: np array of strings. '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1]) #populating feature names in case we don't get alternatively
        else:
            self.feature_names = feature_names

        self.root = self.build_tree(X,y) #calling the function below

    def build_tree(self,X,y):
        '''
        Build the tree recursively, calling in our Branches class for every split.

        INPUT:
            -X: numpy array of our X features data
            -y: numpy array of our y classifications (survived or died)
        OUTPUT:
            Branches object.
        '''

        branch = Branches()
        index, value, new_branches = self.choose_split_index(X,y) #function defined below

        if index is None or len(np.unique(y)) == 1:
            branch.leaf = True
            branch.classes = Counter(y)
            branch.name = branch.classes.most_common(1)[0][0] #returns the Survived or Died based on which happened more often on this leaf.
        else:
            X1,y1, X2,y2 = new_branches
            branch.column = index
            branch.name = self.feature_names[index]
            branch.value = value
            branch.left = self.build_tree(X1,y1)
            branch.right = self.build_tree(X2,y2) #recursively calls itself to make more branches.
        return branch

    def entropy(self,y):
        '''
        calculates the entropy for use in finding the information gain on any
        particular split. That way, we can pick the branches with the highest info gain.
        find notes on the math here: http://www.saedsayad.com/decision_tree.htm


        INPUT: Y or our survived/died classifications
        OUTPUT: float value for entropy.
        '''
        n = y.shape[0] # number of y values
        summed = 0
        for i in np.unique(y):
            pi = sum(y == i)/float(n)
            summed += pi*np.log2(pi)
        return 1-summed

    def make_split(self, X,y,split_index,split_value):
        '''
        Here we create the two subsets of data after we've determined where to split
        the data. We'll call this function later in choose_split_index, which is called
        again in build_tree (so many functions within functions its NUTS).

        INPUT:
            -X: our X data
            -y: our y classifications
            -split_index: integer of which feature COLUMN we're splitting on.
            -split_value: the ROW on that feature that we're splitting on.
        OUTPUT:
            -x1: numpy array of new x's (subset 1)
            -y1: labels for subset 1
            -X2: ... subset 2
            -y2: ... subset 2

        '''

        idx = X[:,split_index] == split_value
        return X[idx],y[idx], X[idx==False],y[idx==False]

    def info_gain(self,y,y1,y2):

        '''
        calculating the info gain for each split. Info gain tells us how much value each
        split has to makeing a better prediction of the outcome, so we'll always pick the
        branch split that has the highest info gain.

        INPUT:
        -y: our original data from the parent branch.
        -y1: subset 1
        -y2: subest 2

        We'll be testing the subsets vs. the parent to see how much info we learned.

        OUTPUT:
        Float value for our info gain.
        Notes on the math here: http://www.saedsayad.com/decision_tree.htm
        '''

        n=y.shape[0]
        parent_info = self.entropy(y)
        new_branch_info = 0
        for yi in (y1,y2):
            new_branch_info += self.entropy(yi)*yi.shape[0]/float(n)
        return parent_info - new_branch_info

    def choose_split_index(self,X,y):
        '''
        Here we're choosing which index to split on and plugging the results into our
        previous function. We want to make sure we pick the split with the highest
        information gain.

        Input:
            -X: our X data
            -y: our y classifications
        OUTPUT:
            -index: integer of which feature to split on
            -value: int of which value to split on
            -splits: (X1 subset, y1 subset, X2 subset, y2subset)
            '''

        split_index = None
        split_value = None
        splits = None
        old_gain = 0
        for x in xrange(X.shape[1]): #for every column
            column = np.unique(X[:,x])
            if len(column)<1:
                continue
            for value in column:
                X1,y1, X2, y2 = self.make_split(X, y, x, value) #told you we'd see this again.
                new_gain = self.info_gain(y,y1,y2)
                if new_gain>old_gain: #only happens if the information gain is higher than the previous splits
                    split_index = x
                    split_value = value
                    splits = (X1,y1,X2,y2)
                    old_gain = new_gain
        return split_index, split_value, splits #return where to split the data according to the highest info_gain

    def predict(self, X):
        '''
        predicting a string of y classifications (Survived or died) based on our model.
         INPUT:
         -X: our NEW X data.
         OUTPUT:
         -y: Predicted values for each datapoint.
         '''
        return np.array([self.root.predict_one(x) for x in X]) #We made this predict_one function already!
