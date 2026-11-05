import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

def clustering_metrics(true, predicted):
    true, predicted = np.asarray(true).reshape(-1), np.asarray(predicted).reshape(-1)
    if len(true) != len(predicted) or not len(true):
        raise ValueError('non-empty label arrays of equal length required')
    table = contingency_matrix(true, predicted)
    rows, cols = linear_sum_assignment(-table)
    return {'ACC': float(table[rows, cols].sum() / len(true)),
            'NMI': float(normalized_mutual_info_score(true, predicted, average_method='arithmetic')),
            'ARI': float(adjusted_rand_score(true, predicted)),
            'PUR': float(table.max(axis=0).sum() / len(true))}
