# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import numpy as np
import sys

import time

from PyRetri.pyretri.config import get_defaults_cfg, setup_cfg
from PyRetri.pyretri.index import build_index_helper, feature_loader
from PyRetri.pyretri.evaluate import build_evaluate_helper


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default=None, metavar='FILE', type=str, help='path to config file')
    args = parser.parse_args()
    return args


def main(original_query, original_gallery, config, k = 50, range = None):

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, config)

    # load features
    start_time = time.time()
    #query_fea, query_info = feature_loader.load_features(original_query)
    #gallery_fea, gallery_info= feature_loader.load_features(original_gallery, range)
    #query_fea = original_query['feature']
    query_fea = original_query
    '''if(range is not None):
        gallery_fea = original_gallery['feature'][range[0]:min(range[1], len(original_gallery['feature']))]
    else:
        gallery_fea = original_gallery['feature']'''
    gallery_fea = original_gallery
    print("Feature Loading Done in: ",time.time()-start_time)
    start_time = time.time()


    index_helper = build_index_helper(cfg.index)
    print("Building index Done in:", time.time() - start_time)
    index_result_info, query_fea, gallery_fea = index_helper.do_index(query_fea, gallery_fea)
    if range is not None:
        index_result_info += range[0]
    print("Overall Indexing done in:", time.time() - start_time)


    return np.array(index_result_info)[:,:k].astype("uint32")

    # build helper and evaluate results
    #evaluate_helper = build_evaluate_helper(cfg.evaluate)
    #mAP, recall_at_k = evaluate_helper.do_eval(index_result_info, gallery_info, gallery_fea, query_fea)

    # show results
    #evaluate_helper.show_results(mAP, recall_at_k)


if __name__ == '__main__':
    main()
