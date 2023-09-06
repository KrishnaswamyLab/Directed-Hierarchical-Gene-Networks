from node2vec import Node2Vec
import numpy as np
import networkx as nx
import pandas as pd
import json
import os

def run_node2vec(data, args):
    
    G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol', create_using=nx.DiGraph)
    node2vec = Node2Vec(G, dimensions=args.dim, walk_length=args.walk_length, num_walks=args.num_walks)
    model = node2vec.fit()
    node_ids = model.wv.index_to_key  # list of node IDs
    embedding = model.wv.vectors

    if not os.path.exists(f'results/{args.model}/{args.dataset}'):
        os.makedirs(f'results/{args.model}/{args.dataset}')
        
    np.savez_compressed(f'results/{args.model}/{args.dataset}/{args.save_as}_results.npz',
                        embedding=embedding,
                        config=vars(args),
                        names=G.nodes)
