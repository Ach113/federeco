from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense


from config import *


def collaborative_filtering_model(num_users: int, num_items: int) -> Model:
    params = MODEL_PARAMETERS['FedNCF']

    mf_dim = params['mf_dim']
    layers = params['layers']
    reg_layers = params['reg_layers']
    reg_mf = params['reg_mf']

    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer='uniform', embeddings_regularizer=l2(reg_mf),
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer='uniform', embeddings_regularizer=l2(reg_mf),
                                  input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)

    return model
