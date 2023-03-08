import pandas as pd
import networkx as nx
import numpy as np
import scipy, torch, keras, json
from keras import layers
from torch_geometric.data import Data

def run_ae(data, args):
    
    keras.utils.set_random_seed(args.seed)
    
    # encoder
    input = keras.Input(shape=(data.shape[1]))
    encoded = layers.Dense(args.dim * 2, activation=args.act, use_bias=args.bias)(input)
    for i in range(args.num_layers - 2):
        encoded = layers.Dense(args.dim * 2, activation=args.act, use_bias=args.bias)(encoded)
        if args.dropout > 0:
            encoded = layers.Dropout(args.dropout)(encoded)

    encoded = layers.Dense(args.dim, activation='linear', use_bias=args.bias)(encoded)

    # decoder
    decoded = layers.Dense(args.dim * 2, activation=args.act,  use_bias=args.bias)(encoded)
    for i in range(args.num_layers - 2):
        decoded = layers.Dense(args.dim * 2, activation=args.act,  use_bias=args.bias)(decoded)
    decoded = layers.Dense(data.shape[1], activation='linear', use_bias=args.bias)(decoded)

    # autoencoder
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr, decay=args.weight_decay), 
                        loss='mean_squared_error')

    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience)

    history = autoencoder.fit(data, data,
                    validation_split = args.val_prop,
                    epochs=args.epochs,
                    shuffle=True,
                    callbacks=[callback])

    data_ae = encoder(data).numpy()
    
    if not os.path.exists(f'results/{args.model}/'):
        os.makedirs(f'results/{args.model}')

    np.save(f'results/{args.model}/{args.save_as}_{args.dataset}_embedding.npy', data_ae)
    with open(f'results/{args.model}/{args.save_as}_{args.dataset}_config.json', 'w') as f:
        json.dump(vars(args), f)
