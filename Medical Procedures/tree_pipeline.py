import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import math
import scipy.stats
from importlib import reload

from clintraj_qi import *
from clintraj_optiscale import *
from clintraj_eltree import *
from clintraj_util import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import skdim

from elpigraph import computeElasticPrincipalTree


class DataForTree:
    def __init__(self, _tree, _pca, _numpy_orig, _components,
                 _mean_val, _var_names):
        self.tree = _tree
        self.pca = _pca
        self.numpy_orig = _numpy_orig
        self.components = _components
        self.mean_val = _mean_val
        self.var_names = _var_names

    def to_dict(self):
        return {
            'tree': self.tree,
            'pca': self.pca,
            'numpy_orig': self.numpy_orig,
            'components': self.components,
            'mean_val': self.mean_val,
            'var_names': self.var_names
        }


def estimate_dimensions(df):
    fisher_s = skdim.id.FisherS().fit(df)
    print("FisherS is ready")

    return fisher_s.dimension_


def do_PCA(X, reduced_dimension):
    X_to_PCA = scipy.stats.zscore(X)
    pca = PCA(n_components=X_to_PCA.shape[1], svd_solver='full')
    Y = pca.fit_transform(X_to_PCA)
    v = pca.components_.T
    mean_val = np.mean(X_to_PCA, axis=0)
    return Y[:, 0:reduced_dimension], v, mean_val


def create_extended_tree(X, nnodes=50):
    tree_elpi = \
        computeElasticPrincipalTree(X, nnodes, alpha=0.01, Mu=0.1, Lambda=0.05, FinalEnergy='Penalized', n_cores=-1)[0]
    prune_the_tree(tree_elpi)
    tree_extended = ExtendLeaves_modified(X, tree_elpi, Mode="QuantDists", ControlPar=.5, DoSA=False)
    return tree_extended


def plot_trees(df: DataForTree, main_feature, other_features, figsize=(15, 10)):
    fig3 = plt.figure(figsize=figsize, constrained_layout=True)

    gs = fig3.add_gridspec(len(other_features), 1 + len(other_features))

    f3_ax1 = fig3.add_subplot(gs[:, :-1])
    visualize_eltree_with_data(df.tree, df.pca, df.numpy_orig,
                               df.components, df.mean_val, 'k', df.var_names,
                               Color_by_feature=main_feature, cmap='gist_rainbow', add_color_bar=True,
                               Transparency_Alpha_points=0.5)
    f3_ax1.set_title(main_feature)

    for num_feature in range(len(other_features)):
        f3_ax = fig3.add_subplot(gs[num_feature, -1])
        feature = other_features[num_feature]
        cmap = 'winter' if (feature == 'Sex') else 'gist_rainbow'
        visualize_eltree_with_data(df.tree, df.pca, df.numpy_orig,
                                   df.components, df.mean_val, 'k', df.var_names,
                                   Color_by_feature=feature, cmap=cmap, add_color_bar=True,
                                   Transparency_Alpha_points=0.5)
        f3_ax.set_title(feature)

    plt.show()


def plot_tree_pipeline(data_encoded_numpy, data_numeric, main_feature, other_features):
    print(f"Estimating dimensions")
    dimensions = estimate_dimensions(data_encoded_numpy)
    print("Estimating done!\n\n")

    pca, components, mean_val = do_PCA(data_encoded_numpy, math.ceil(dimensions))
    tree = create_extended_tree(pca)
    data_for_tree = DataForTree(
        tree,
        pca,
        data_numeric.fillna(data_numeric.median()).to_numpy(),
        components,
        mean_val,
        [str(s) for s in data_numeric.columns[:]]
    )

    plot_trees(data_for_tree, main_feature, other_features)

    return data_for_tree.to_dict()
