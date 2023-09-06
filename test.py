from config import parser
import numpy as np
import pandas as pd
import torch
import json
from run.run_node2vec import run_node2vec
from run.run_magnet import run_magnet
from run.run_kge import run_kge
from run.run_gae import run_gae
from run.run_ae import run_ae
from run.run_directed_scattering import get_pretrained_directed_scattering, run_directed_scattering
from run.run_undirected_scattering import get_pretrained_undirected_scattering, run_undirected_scattering
from run.run_pm import run_pm

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.dataset == 'omnipath':
        splits = np.load(f'data/{args.dataset}_curated_interactions_with_edges.npz')
        data = np.vstack((splits['train'], splits['val'])) # keep training and validation graph only
        data = pd.DataFrame(data, columns=['source_genesymbol','type', 'target_genesymbol'])
    else:
        splits = np.load(f'data/{args.dataset}_curated_interactions.npz')
        data = np.vstack((splits['train'], splits['val'])) # keep training and validation graph only
        data = pd.DataFrame(data, columns=['source_genesymbol', 'target_genesymbol'])

    if args.model == 'Node2Vec':
        run_node2vec(data, args)

    elif args.model == 'MagNet':
        run_magnet(data, args)
        
    elif args.model == 'TransE':
        run_kge(data, args)

    elif args.model == 'GAE':
        run_gae(data, args)
     
    elif args.model == 'DS-AE':
        if args.compute_scattering:
            run_directed_scattering(data, args)
            
        ds = get_pretrained_directed_scattering(data, args)
        run_ae(ds, args)
        
    elif args.model == 'UDS-AE':
        if args.compute_scattering:
            run_undirected_scattering(data, args)
            
        uds = get_pretrained_undirected_scattering(data, args)
        run_ae(uds, args)
    
    elif args.model == 'DS-PM':
        if args.compute_scattering:
            run_directed_scattering(data, args)
            
        ds = get_pretrained_directed_scattering(data, args)
        run_pm(ds, args)

    elif args.model == 'HypMagNet':
        run_hypmagnet(data, args)

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
