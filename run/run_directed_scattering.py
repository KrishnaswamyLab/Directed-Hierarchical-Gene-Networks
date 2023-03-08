import sys
sys.path.append('/home/aarthivenkat/cell_cell_comm/graph_scattering_FINAL')

import pandas as pd
import torch
import networkx as nx
import numpy as np
import graphscattering as gs
import time
from sklearnex import patch_sklearn
patch_sklearn()

def get_pretrained_directed_scattering(data, args):
    q = float(args.q)
    J = 15
    ds = np.load(f'results/{args.model}/Directed_Scattering_J{J}_q{q}_{args.dataset}_embedding.npy')
    return ds

def run_directed_scattering(data, args):
    G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol', create_using=nx.DiGraph)
    A = nx.adjacency_matrix(G).toarray()

    N = A.shape[0]
    signal = np.random.randn(N, 1)
    q = args.q
    J = 15
    vals, vecs = gs.compute_eigen(A, q)
    scales = np.flip(np.arange(0, J+1))
    all_features = gs.compute_all_features(vals, vecs, signal, N, "lowpass", scales)
    
    if not os.path.exists(f'results/{args.model}/'):
        os.makedirs(f'results/{args.model}')
    
    filename = f"results/{args.model}/Directed_Scattering_J{J}_q{q}_{args.dataset}_embedding.npy"
    np.save(filename, all_features)
