import numpy as np
import pandas as pd
from collections import Counter

class Branches(object):
    '''
    A branches class for a decision tree. This is where each split in the data is
    going to be contained as the tree forms. They will eventually lead to a leaf
    where our data is either categorized as Survived or Died.
    '''

    def __init__(self):
        self.name = None    #this is the name of the X feature we're splitting on
        self.column = None  #this is which X feature to split on
        self.value = None   #The value where we place our split (i.e the split location)
        self.leaf = False   # make True if this split is a leaf (end of split).
        self.left = None    #One side of the split.
        self.right = None   #the other
        self.classes = Counter() #for the leaves, the key is the Surived or died,
                                #the value is the count of data points in that key.

    def predict_one(self,X):
        '''
        This function will predict the outcome for any particular datapoint.

        INPUT: 1 dimension np array
        OUTPUT: Survived or Died label.
        '''

        if self.leaf == True: #will only occur on the last branch.
            return self.name
        col_value = X[self.column]

        if col_value == self.value:
            return self.left.predict_one(X)
        else:
            return self.right.predict_one(X) #Recursively calls itself until we hit the leaf.
