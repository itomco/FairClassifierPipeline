#Import required libraries
import pandas as pd
import numpy as np

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.base import BaseEstimator, ClassifierMixin

import rrcf

class RobustRandomCutForest(ClassifierMixin, BaseEstimator):
    """
    RobustRandomCutForest.
    The Robust Random Cut Forest (RRCF) algorithm is an ensemble method for detecting outliers in streaming data. RRCF offers a number of features that many competing anomaly detection algorithms lack. Specifically, RRCF:
    Is designed to handle streaming data.
    Performs well on high-dimensional data.
    Reduces the influence of irrelevant dimensions.
    Gracefully handles duplicates and near-duplicates that could otherwise mask the presence of outliers.
    Features an anomaly-scoring algorithm with a clear underlying statistical meaning.
    This repository provides an open-source implementation of the RRCF algorithm and its core data structures for the purposes of facilitating experimentation and enabling future extensions of the RRCF algorithm.
    Parameters
    ----------
    num_trees : int, default=100
        The number of trees.
    tree_size : int, default=512
        The maximum depth of each tree

    Attributes
    ----------

    avg_cod : Compute average CoDisp
              Explanation about CoDisp:
              The likelihood that a point is an outlier is measured by its collusive displacement
              (CoDisp): if including a new point significantly changes the model complexity (i.e. bit depth),
              then that point is more likely to be an outlier.
              tree.codisp('inlier') >>> 1.75
              tree.codisp('outlier') >>> 39.0

    Notes
    -----
    https://github.com/kLabUM/rrcf

    """

    def __init__(self, *, num_trees=100, tree_size=512, contamination:float=0.0):
        # print('>> Init RobustRandomCutForest')
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.contamination = contamination

    def fit(self, X, y=None):
        # do nothing :-)
        # Return the classifier
        return self

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"num_trees": self.num_trees,
                "tree_size": self.tree_size,
                "contamination": self.contamination}



    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        n = len(X)
        forest = []
        # build trees for RRCF
        while len(forest) < self.num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n,
                                   size=(1, self.tree_size),  # size=(n // tree_size, tree_size),
                                   replace=True)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)

        # Compute average CoDisp
        avg_codisp_d = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp_d[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1.0)
        avg_codisp_d /= index

        mask=np.percentile(avg_codisp_d,int((1-self.contamination)*100.0))
        avg_codisp_d[avg_codisp_d <= mask] = 1
        avg_codisp_d[avg_codisp_d > mask] = -1

        return avg_codisp_d


