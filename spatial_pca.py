import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


def autocorrelation(df_t, neighbourhood, feat):
    xm = np.mean(df_t[feat])

    W = np.sum([len(ns) for cid, ns in neighbourhood.items()])
    N = len(df_t)
    norm = np.sum([(v - xm)**2 for v in df_t[feat]])

    s = 0
    cids = set(df_t["id"])

    for cid in df_t["id"]:
        for n in neighbourhood[cid]:
            nid = n[0]
            if nid in cids:
                wc = (df_t[df_t["id"] == nid][feat] - xm)
                wn = (df_t[df_t["id"] == cid][feat] - xm)

            s = s + (wc.values[0] * wn.values[0])

    return (N/W) * (s / norm)


def get_adj(neighs):
    G = nx.Graph()
    edges = [(cid, n[0]) for cid, ns in neighs.items() for n in ns]

    G.add_edges_from(edges)
    A = nx.adjacency_matrix(G).toarray()
    sumA = np.sum(A, axis=1)

    L = A / sumA[:, None]

    return L


def get_criterion(feats, neighs):
    L = get_adj(neighs)
    X = feats.to_numpy()

    n = len(X)

    M = np.transpose(X)@(L+np.transpose(L))@X

    M = M / (2*n)

    
    return X, M


def red_spca(feats, neighs, n_comps=2):
    X, M = get_criterion(feats, neighs)

    eivals, eivectors = np.linalg.eig(M)
    eivectors = eivectors[:, np.argsort(eivals)[::-1]]

    eivs = eivectors

    X_ = X@eivs[:, [0, 1]]

    return X_, eivals

