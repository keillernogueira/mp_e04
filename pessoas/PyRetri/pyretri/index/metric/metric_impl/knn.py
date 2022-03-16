# -*- coding: utf-8 -*-

import torch

from ..metric_base import MetricBase
from ...registry import METRICS

from typing import Dict

import numpy as np

@METRICS.register
class KNN(MetricBase):
    """
    Similarity measure based on the euclidean distance.

    Hyper-Params:
        top_k (int): top_k nearest neighbors will be output in sorted order. If it is 0, all neighbors will be output.
    """
    default_hyper_params = {
        "top_k": 0,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(KNN, self).__init__(hps)

    def _cal_dis(self, query_fea: torch.tensor, gallery_fea: torch.tensor, device: str) -> torch.tensor:
        """
        Calculate the distance between query set features and gallery set features.

        Args:
            query_fea (torch.tensor): query set features.
            gallery_fea (torch.tensor): gallery set features.
            device (str): Device to be utilized by pytorch. 

        Returns:
            dis (torch.tensor): the distance between query set features and gallery set features.
        """
        query_fea = query_fea.to(device)
        gallery_fea = gallery_fea.to(device)
        query_fea = query_fea.transpose(1, 0)
        inner_dot = gallery_fea.mm(query_fea)

        '''
        #Uncoment this section to reduce GPU memory usage in retrieval process.
        #This changes the distance metric from euclidean distance to cosine,
        #removing the memory cost associated with squaring the features matrix.

        inner_dot = inner_dot.transpose(1,0)
        return 1 - inner_dot
        return np.array(inner_dot.cpu())

        '''

        dis = (gallery_fea ** 2).sum(dim=1, keepdim=True) + (query_fea ** 2).sum(dim=0, keepdim=True)
        dis = dis - 2 * inner_dot
        dis = dis.transpose(1, 0)

        return dis

    def __call__(self, query_fea: torch.tensor, gallery_fea: torch.tensor, device: str) -> (torch.tensor, torch.tensor):

        dis = self._cal_dis(query_fea, gallery_fea, device)
        #return dis
        sorted_index = torch.argsort(dis, dim=1)
        if self._hyper_params["top_k"] != 0:
            sorted_index = sorted_index[:, :self._hyper_params["top_k"]]
        return dis, sorted_index
