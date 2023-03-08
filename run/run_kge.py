import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import networkx as nx
import numpy as np
import json

def run_kge(data, args):
    if args.edge_attribute == 0:
        data['type'] = 'OmniPath'
        
    tf = TriplesFactory.from_labeled_triples(data[['source_genesymbol', 'type', 'target_genesymbol']].values)
    training, testing, validation = tf.split([1-(args.test_prop + args.val_prop), args.test_prop, args.val_prop], random_state=args.seed)

    # don't need graph, just order of nodes
    G = nx.from_pandas_edgelist(data, source='source_genesymbol', target='target_genesymbol', create_using=nx.DiGraph)
    entities = G.nodes()
    del(G)

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=args.model,
        stopper='early',
        optimizer='Adam',
        random_seed=args.seed,
        loss="NSSA",
        device='cpu',
        model_kwargs ={'embedding_dim':args.dim},
        training_kwargs=dict(num_epochs=args.epochs),
        stopper_kwargs=dict(patience=args.patience, frequency=1),
        optimizer_kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
        loss_kwargs=dict(adversarial_temperature=args.temperature, margin=args.margin), 
    )
    
    model_path = f'results/{args.model}/{args.save_as}_{args.dataset}_model.pkl'
    result.save_model(model_path)    
    model = torch.load(model_path)
    
    embedding = model.entity_representations[0](torch.LongTensor(tf.entities_to_ids(entities)))
    
    if not os.path.exists(f'results/{args.model}/'):
        os.makedirs(f'results/{args.model}')
    
    np.save(f'results/{args.model}/{args.save_as}_{args.dataset}_embedding.npy', embedding.detach().cpu().numpy())
    with open(f'results/{args.model}/{args.save_as}_{args.dataset}_config.json', 'w') as f:
        json.dump(vars(args), f)
