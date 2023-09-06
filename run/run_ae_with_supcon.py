import keras
import tensorflow as tf
from keras import layers
import tensorflow_addons as tfa
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

"""
From arXiv:2004.11362 and https://keras.io/examples/vision/supervised-contrastive-learning/
"""
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
        
def run_ae_with_supcon(data, y1, y2, seed=1234, act='relu', bias=1, dim=128, num_layers=2, dropout=0.0, lr=0.001, epochs=100, 
           val_prop=0.05, weight_decay=0, patience=10):
    
    keras.utils.set_random_seed(seed)
    
    # encoder
    input = keras.Input(shape=(data.shape[1]))
    encoded = layers.Dense(dim * 2, activation=act, use_bias=bias)(input)
    for i in range(num_layers - 2):
        encoded = layers.Dense(dim * 2, activation=act, use_bias=bias)(encoded)
        if dropout > 0:
            encoded = layers.Dropout(dropout)(encoded)

    encoded = layers.Dense(dim, activation='linear', use_bias=bias)(encoded)
    classifier1 = layers.Dense(dim, activation='linear', name='classifier1')(encoded)
    classifier2 = layers.Dense(dim, activation='linear', name='classifier2')(encoded)

    # decoder
    decoded = layers.Dense(dim * 2, activation=act,  use_bias=bias)(encoded)
    for i in range(num_layers - 2):
        decoded = layers.Dense(dim * 2, activation=act,  use_bias=bias)(decoded)
    decoded = layers.Dense(data.shape[1], activation='linear', use_bias=bias, name='decoder')(decoded)

    # autoencoder
    autoencoder = keras.Model(input, [decoded, classifier1, classifier2])
    encoder = keras.Model(input, encoded)
    autoencoder.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr, decay=weight_decay), 
                        loss=['mean_squared_error', SupervisedContrastiveLoss(1), SupervisedContrastiveLoss(1)],
                        loss_weights=[10,5,1])

    callback = keras.callbacks.EarlyStopping(monitor="classifier2_loss", patience=patience)

    history = autoencoder.fit(data, [data,y1,y2],
                    epochs=epochs,
                    shuffle=True,
                    callbacks=[callback])

    embedding = encoder(data).numpy()
    
    return (embedding)