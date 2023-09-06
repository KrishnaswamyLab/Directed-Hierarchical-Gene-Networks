import tensorflow as tf
from tensorflow import keras
from hyperlib.nn.layers.lin_hyp import LinearHyperbolic
from hyperlib.nn.optimizers.rsgd import RSGD
from hyperlib.manifold.poincare import Poincare
import numpy as np
import json
from scipy import stats
import tensorflow as tf
import os

def run_pm(data, args):
    
    names = data.index
    data = data.values # convert dataframe to array

    tf.random.set_seed(args.seed)
    
    manifold = Poincare()
    
    # Create layers
    lambda_layer_1 = tf.keras.layers.Lambda(lambda x: manifold.expmap0(x, c=args.c))
    hyperbolic_layer_1 = LinearHyperbolic(args.dim*2, manifold, args.c, activation=args.act, use_bias=args.bias)
    embedding_layer = LinearHyperbolic(args.dim, manifold, args.c, activation='linear', use_bias=args.bias)    
    hyperbolic_layer_2 = LinearHyperbolic(data.shape[1], manifold, args.c, activation='linear', use_bias=args.bias)    
    lambda_layer_2 = tf.keras.layers.Lambda(lambda x: manifold.logmap0(x, c=args.c))
    
    # Create optimizer
    optimizer = RSGD(learning_rate=args.lr, decay=args.weight_decay)

    # Create model architecture
    autoencoder = tf.keras.models.Sequential([
      lambda_layer_1,
      hyperbolic_layer_1,
      embedding_layer,
      hyperbolic_layer_2,
      lambda_layer_2
    ])

    encoder = tf.keras.models.Sequential([
      lambda_layer_1,
      hyperbolic_layer_1,
      embedding_layer,
    ])

    # Compile the model with the Riemannian optimizer            
    autoencoder.compile(
        optimizer=optimizer,
        loss='mse')

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.patience)
    history = autoencoder.fit(data, data, epochs = args.epochs,
                              shuffle=True, callbacks=[callback])

    data_ae = encoder(data).numpy()

    if not os.path.exists(f'results/{args.model}/{args.dataset}'):
        os.makedirs(f'results/{args.model}/{args.dataset}')
        
    np.savez_compressed(f'results/{args.model}/{args.dataset}/{args.save_as}_results.npz',
                        embedding=data_ae,
                        config=vars(args),
                        names=names)