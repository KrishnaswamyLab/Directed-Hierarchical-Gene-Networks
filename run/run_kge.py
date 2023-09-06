import os
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import networkx as nx
import numpy as np
import json

def run_kge(data, args):
    if args.edge_attribute == 0:
        data['type'] = args.dataset
    
    train = TriplesFactory.from_labeled_triples(data[['source_genesymbol', 'type', 'target_genesymbol']].astype(str).values)
    
    # don't need graph, just order of nodes
    G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol', create_using=nx.DiGraph)
    entities = [str(x) for x in G.nodes()]

    result = pipeline(
        training=train,
        testing=train,
        model=args.model,
        optimizer='Adam',
        random_seed=args.seed,
        loss="NSSA",
        device='cpu',
        model_kwargs ={'embedding_dim':args.dim},
        training_kwargs=dict(num_epochs=args.epochs),
        optimizer_kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
        loss_kwargs=dict(adversarial_temperature=args.temperature, margin=args.margin), 
    )
    
    if not os.path.exists(f'results/{args.model}/{args.dataset}'):
        os.makedirs(f'results/{args.model}/{args.dataset}')
        
    model_path = f'results/{args.model}/{args.dataset}/{args.save_as}_{args.dataset}_model.pkl'
    result.save_model(model_path)    
    model = torch.load(model_path)
    
    embedding = model.entity_representations[0](torch.LongTensor(train.entities_to_ids(entities)))
        
    np.savez_compressed(f'results/{args.model}/{args.dataset}/{args.save_as}_results.npz',
                        embedding=embedding.detach().cpu().numpy(),
                        config=vars(args),
                        names=entities)
