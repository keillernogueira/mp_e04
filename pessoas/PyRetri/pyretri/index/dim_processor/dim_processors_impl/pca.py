# -*- coding: utf-8 -*-

import numpy as np

from ..dim_processors_base import DimProcessorBase
from ...registry import DIMPROCESSORS
from ...utils import feature_loader
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as SKPCA

import sys
import scipy.io

from typing import Dict, List

@DIMPROCESSORS.register
class PCA(DimProcessorBase):
    """
    Do the PCA transformation for dimension reduction.

    Hyper-Params:
        proj_dim (int): the dimension after reduction. If it is 0, then no reduction will be done.
        whiten (bool): whether do whiten.
        train_fea_dir (str): the path of features for training PCA.
        l2 (bool): whether do l2-normalization for the training features.
    """
    default_hyper_params = {
        "proj_dim": 0,
        "whiten": True,
        "train_fea_dir": "unknown",
        "l2": True,
    }

    def __init__(self, feature_names: List[str], hps: Dict or None = None):
        """
        Args:
            feature_names (list): a list of features names to be loaded.
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(PCA, self).__init__(feature_names, hps)

        self.pca = SKPCA(n_components=self._hyper_params["proj_dim"], whiten=self._hyper_params["whiten"], svd_solver = 'arpack')
        self._train(self._hyper_params["train_fea_dir"])

    def _train(self, fea_dir: str) -> None:
        """
        Train the PCA.

        Args:
            fea_dir (str): the path of features for training PCA.
        """
        train_fea = scipy.io.loadmat(sys.path[0] + '/' + fea_dir)
        train_fea = train_fea['feature']
        print(train_fea.shape)
        if self._hyper_params["l2"]:
            train_fea = normalize(train_fea, norm="l2")
        print(train_fea.shape)
        self.pca.fit(train_fea)

    def __call__(self, fea: np.ndarray) -> np.ndarray:
        ori_fea = fea
        print("pre transform")
        print(ori_fea.shape)
        print(self.pca)
        proj_fea = self.pca.transform(ori_fea)
        print("transformou")
        print(proj_fea)
        return proj_fea


