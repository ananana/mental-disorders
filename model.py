
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Lambda, BatchNormalization, TimeDistributed, \
    Bidirectional, Input, concatenate, Flatten, RepeatVector, Activation, Multiply, Permute, \
    Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
def build_hierarchical_model(hyperparams, hyperparams_features, embedding_matrix, emotions, stopwords_list,
                liwc_categories,
               ignore_layer=[], activations=None, classes=1):
    def attention(xin):
        return K.sum(xin, axis=1) 


    # Post/sentence representation - word sequence
    tokens_features = Input(shape=(hyperparams['maxlen'],), name='word_seq')
    embedding_layer = Embedding(hyperparams_features['max_features'], 
                                hyperparams_features['embedding_dim'], 
                                input_length=hyperparams['maxlen'],
                                embeddings_regularizer=regularizers.l2(hyperparams['l2_embeddings']),
                                weights=[embedding_matrix], 
                                trainable=hyperparams['trainable_embeddings'],
                               name='embeddings_layer')(
        tokens_features)
    embedding_layer = Dropout(hyperparams['dropout'], name='embedding_dropout')(embedding_layer)

    
    if 'lstm' not in ignore_layer:
        if False: #tf.test.is_gpu_available():
            lstm_layers = CuDNNLSTM(hyperparams['lstm_units'], 
                                    return_sequences='attention' not in ignore_layer, # only True if using attention
                          name='LSTM_layer')(embedding_layer)
        else:
            lstm_layers = LSTM(hyperparams['lstm_units'], 
                               return_sequences='attention' not in ignore_layer,
                          name='LSTM_layer')(embedding_layer)

        # Attention
        if 'attention' not in ignore_layer:
            attention_layer = Dense(1, activation='tanh', name='attention')
            attention = attention_layer(lstm_layers)
            attention = Flatten()(attention)
            attention_output = Activation('softmax')(attention)
            attention = RepeatVector(hyperparams['lstm_units'])(attention_output)
            attention = Permute([2, 1])(attention)

            sent_representation = Multiply()([lstm_layers, attention])
            sent_representation = Lambda(lambda xin: K.sum(xin, axis=1), 
                                     output_shape=(hyperparams['lstm_units'],)
                                    )(sent_representation)

#             sent_representation = Lambda(attention, 
#                                          output_shape=(hyperparams['lstm_units'],
#                                         ))(sent_representation)
        else:
            sent_representation = lstm_layers

    elif 'cnn' not in ignore_layer:
        cnn_layers = Conv1D(hyperparams['filters'],
                             hyperparams['kernel_size'],
                             padding='valid',
                             activation='relu',
                             strides=1)(embedding_layer)
        # we use max pooling:
        sent_representation = GlobalMaxPooling1D()(cnn_layers)
    
    
    if 'batchnorm' not in ignore_layer:
        sent_representation = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                          name='sent_repr_norm')(sent_representation)
    sent_representation = Dropout(hyperparams['dropout'], name='sent_repr_dropout')(sent_representation)

            # Other features 
    numerical_features_history = Input(shape=(
            hyperparams['posts_per_group'],
            len(emotions) + 1 + len(liwc_categories)
        ), name='numeric_input_hist') # emotions and pronouns
    sparse_features_history = Input(shape=(
            hyperparams['posts_per_group'],
            len(stopwords_list)
        ), name='sparse_input_hist') # stopwords



    if activations == 'attention':
#         sent_representation = Flatten()(attention_layer.output)
        sent_representation = attention_output


    posts_history_input = Input(shape=(hyperparams['posts_per_group'], 
                                     hyperparams['maxlen']
                                          ), name='hierarchical_word_seq_input')

    # Hierarchy
    sentEncoder = Model(inputs=tokens_features, 
                        outputs=sent_representation)
    sentEncoder.summary()

    user_encoder = TimeDistributed(sentEncoder, name='user_encoder')(posts_history_input)



    
    if activations != 'attention':
        
        
        # BERT encoder
        if 'bert_layer' not in hyperparams['ignore_layer']:
            in_id_bert = Input(shape=(hyperparams['maxlen'],), name="input_ids_bert")
            in_mask_bert = Input(shape=(hyperparams['maxlen'],), name="input_masks_bert")
            in_segment_bert = Input(shape=(hyperparams['maxlen'],), name="segment_ids_bert")
            bert_inputs = [in_id_bert, in_mask_bert, in_segment_bert]

            bert_output = BertLayer(n_fine_tune_layers=hyperparams['bert_finetune_layers'], 
                                    pooling=hyperparams['bert_pooling'],
                                   trainable=hyperparams['bert_trainable'],
                                   name='bert_layer')(bert_inputs)
            dense_bert = Dense(hyperparams['bert_dense_units'], 
                               activation='relu',
                              kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                              name='bert_dense_layer')(bert_output)

            bertSentEncoder = Model(bert_inputs, dense_bert)


            in_id_bert_history = Input(shape=(hyperparams['posts_per_group'],
                                                              hyperparams['maxlen'],), name="input_ids_bert_hist")
            in_mask_bert_history = Input(shape=(hyperparams['posts_per_group'],
                                                                hyperparams['maxlen'],), name="input_masks_bert_hist")
            in_segment_bert_history = Input(shape=(hyperparams['posts_per_group'],
                                                                   hyperparams['maxlen'],), name="segment_ids_bert_hist")
            bert_inputs_history = [in_id_bert_history, in_mask_bert_history, in_segment_bert_history]
            bert_inputs_concatenated = concatenate(bert_inputs_history)
            inputs_indices = [hyperparams['maxlen']*i for i in range(3)]
            # slice the input in equal slices on the last dimension
            bert_encoder_layer = TimeDistributed(Lambda(lambda x: bertSentEncoder([x[:,inputs_indices[0]:inputs_indices[1]], 
                                                                          x[:,inputs_indices[1]:inputs_indices[2]],
                                                                                  x[:,inputs_indices[2]:]])),
                                                name='bert_distributed_layer')(
                                bert_inputs_concatenated)
            bertUserEncoder = Model(bert_inputs_history, bert_encoder_layer)
            bertUserEncoder.summary()

            bert_user_encoder = bertUserEncoder(bert_inputs_history)
        else:
            bert_user_encoder = None


        dense_layer_sparse = Dense(units=hyperparams['dense_bow_units'],
                                  name='sparse_feat_dense_layer', activation='relu',
                                    kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
        dense_layer_sparse_user = TimeDistributed(dense_layer_sparse,
                                                 name='sparse_dense_layer_user')(sparse_features_history)
        
                
        dense_layer_numerical = Dense(units=hyperparams['dense_numerical_units'],
                                  name='numerical_feat_dense_layer', activation='relu',
                                    kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
        dense_layer_numerical_user = TimeDistributed(dense_layer_numerical,
                                                 name='numerical_dense_layer_user')(numerical_features_history)


        # Concatenate features
        if 'batchnorm' not in ignore_layer:
#             numerical_features_history_norm = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
#                                                          name='numerical_features_norm')(numerical_features_history)
            dense_layer_numerical_user = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                         name='numerical_features_norm')(dense_layer_numerical_user)
            dense_layer_sparse_user = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                         name='sparse_features_norm')(dense_layer_sparse_user)
        all_layers = {
            'user_encoded': user_encoder,
            'bert_layer': bert_user_encoder,
#             'numerical_dense_layer': numerical_features_history if 'batchnorm' in ignore_layer \
#                         else numerical_features_history_norm
            'numerical_dense_layer': dense_layer_numerical_user,

            'sparse_feat_dense_layer': dense_layer_sparse_user,
        }

        layers_to_merge = [l for n,l in all_layers.items() if n not in ignore_layer]
        if len(layers_to_merge) == 1:
            merged_layers = layers_to_merge[0]
        else:
            merged_layers = concatenate(layers_to_merge)

        if 'lstm_user' not in ignore_layer:

            if False:#tf.test.is_gpu_available():
                lstm_user_layers = CuDNNLSTM(hyperparams['lstm_units_user'], 
                                        return_sequences='attention_user' not in ignore_layer, # only True if using attention
                              name='LSTM_layer_user')(merged_layers)
            else:
                lstm_user_layers = LSTM(hyperparams['lstm_units_user'], 
                                   return_sequences='attention_user' not in ignore_layer,
                              name='LSTM_layer_user')(merged_layers)

            # Attention
            if 'attention_user' not in ignore_layer:
                attention_user_layer = Dense(1, activation='tanh', name='attention_user')
                attention_user = attention_user_layer(lstm_user_layers)
                attention_user = Flatten()(attention_user)
                attention_user_output = Activation('softmax')(attention_user)
                attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
                attention_user = Permute([2, 1])(attention_user)

                user_representation = Multiply()([lstm_user_layers, attention_user])
                user_representation = Lambda(lambda xin: K.sum(xin, axis=1), 
                                             output_shape=(hyperparams['lstm_units_user'],))(user_representation)
    #             user_representation = Lambda(attention, 
    #                                          output_shape=(hyperparams['lstm_units_user'],
    #                                         ))(user_representation)
            else:
                user_representation = lstm_user_layers


        elif 'cnn_user' not in ignore_layer:
            cnn_layers_user = Conv1D(hyperparams['filters_user'],
                                 hyperparams['kernel_size_user'],
                                 padding='valid',
                                 activation='relu',
                                 strides=1)(merged_layers)
            # we use max pooling:
            user_representation = GlobalMaxPooling1D()(cnn_layers_user)
    #         user_representation = Flatten()(user_representation)


        user_representation = Dropout(hyperparams['dropout'], name='user_repr_dropout')(user_representation)


        if hyperparams['dense_user_units']:
            user_representation = Dense(units=hyperparams['dense_user_units'], activation='relu',
                                       name='dense_user_representation')(user_representation)

        output_layer = Dense(classes, activation='sigmoid' if classes==1 else 'softmax',
                             name='output_layer',
                            kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                            )(user_representation)

    # Compile model

#     elif activations == 'attention':
#         outputs = attention_layer.output
    if activations == 'attention':
        outputs = user_encoder

        
    elif activations == 'attention_user':
#         outputs = attention_user.output
        outputs = attention_user_output
    elif activations == 'output_layer':
        outputs = user_representation

    
    else:
        outputs = output_layer
    if 'bert_layer' not in hyperparams['ignore_layer']:

        hierarchical_model = Model(inputs=[posts_history_input, 
                                       numerical_features_history, sparse_features_history,
                                      in_id_bert_history, in_mask_bert_history, in_segment_bert_history], 
                  outputs=outputs)
    else:
        hierarchical_model = Model(inputs=[posts_history_input, 
                                       numerical_features_history, sparse_features_history,
                                      ], 
                  outputs=outputs)
    hierarchical_model.summary()
    
    if classes==1:
        metrics_class = Metrics(threshold=hyperparams['threshold'])
        hierarchical_model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                      metrics=['f1', 'precision', 'recall', 'auc'])
    else:
        
        hierarchical_model.compile(hyperparams['optimizer'], K.categorical_crossentropy,
                     metrics=[ tf.keras.metrics.CategoricalAccuracy(name='cat_acc'),
#                              tf.keras.metrics.Precision(name='prec'), tf.keras.metrics.Recall(name='rec'), 
                              ])
    return hierarchical_model
