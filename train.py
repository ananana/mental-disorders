from callbacks import FreezeLayer, WeightsHistory, OutputsHistory, LRHistory, ActivationsAttention
from tensorflow.keras import callbacks
from metrics import Metrics
from comet_ml import Experiment, Optimizer
import logging, sys
import pickle
from DataGenerator import DataGenerator
from model import build_hierarchical_model

def train_model(model, hyperparams,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=4,
                callback_list = [], logger=None,
                
                model_path='/tmp/model',
                validation_set='valid',
               verbose=1):
    
    if not logger:
      logger = logging.getLogger('training')
      ch = logging.StreamHandler(sys.stdout)
      # create formatter
      formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
      # add formatter to ch
      ch.setFormatter(formatter)
      # add ch to logger
      logger.addHandler(ch)
      logger.setLevel(logging.DEBUG)
    logger.info("Initializing callbacks...\n")
    # Initialize callbacks
    freeze_layer = FreezeLayer(patience=hyperparams['freeze_patience'], set_to=not hyperparams['trainable_embeddings'])
    weights_history = WeightsHistory()
    outputs_history_valid = OutputsHistory(generator_type=validation_set)
    outputs_history_train = OutputsHistory(generator_type='train')
    activations_history_train = ActivationsAttention(generator_type='train')
    lr_history = LRHistory()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=hyperparams['reduce_lr_factor'],
                              patience=hyperparams['reduce_lr_patience'], min_lr=0.000001, verbose=1)
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr: 
                                                  lr if (epoch+1)%hyperparams['scheduled_reduce_lr_freq']!=0 else
                                                  lr*hyperparams['scheduled_reduce_lr_factor'], verbose=1)
    callbacks_dict = {'freeze_layer': freeze_layer, 'weights_history': weights_history,
           'outputs_history_valid': outputs_history_valid, 'outputs_history_train': outputs_history_train,
           'lr_history': lr_history,
            'activations': activations_history_train,
           'reduce_lr_plateau': reduce_lr,
            'lr_schedule': lr_schedule}

    
    logging.info('Train...')


    history = model.fit_generator(data_generator_train,
                steps_per_epoch=100,
              epochs=epochs, initial_epoch=start_epoch, 
              class_weight=class_weight,
              validation_data=data_generator_valid,
                        verbose=verbose,
#               validation_split=0.3,
                       workers=workers,
            callbacks = [
                callbacks.ModelCheckpoint(filepath='%s_best.h5' % model_path, verbose=1, 
                                          save_best_only=True, save_weights_only=True),
#                 callbacks.EarlyStopping(patience=hyperparams['early_stopping_patience'],
#                                        restore_best_weights=True)
            ] + [
                callbacks_dict[c] for c in [
#                     'weights_history', 
                    'outputs_history_valid', 
#                     'outputs_history_train', 
#                     'reduce_lr_plateau', 
#                     'lr_schedule', 
#                     'activations'
                ]])
    return model, history

def get_network_type(hyperparams):
    if 'lstm' in hyperparams['ignore_layer']:
        network_type = 'cnn'
    else:
        network_type = 'lstm'
    if 'user_encoded' in hyperparams['ignore_layer']:
        if 'bert_layer' not in hyperparams['ignore_layer']:
            network_type = 'bert'
        else:
            network_type = 'extfeatures'
    if hyperparams['hierarchical']:
        hierarch_type = 'hierarchical'
    else:
        hierarch_type = 'seq'
    return network_type, hierarch_type

def initialize_experiment(hyperparams, nrc_lexicon_path, emotions, pretrained_embeddings_path, 
                          dataset_type, transfer_type, hyperparams_features):

    experiment = Experiment(api_key="eoBdVyznAhfg3bK9pZ58ZSXfv",
                            project_name="mental", workspace="ananana", disabled=False)

    experiment.log_parameters(hyperparams_features)

    experiment.log_parameter('emotion_lexicon', nrc_lexicon_path)
    experiment.log_parameter('emotions', emotions)
    experiment.log_parameter('embeddings_path', pretrained_embeddings_path)
    experiment.log_parameter('dataset_type', dataset_type)
    experiment.log_parameter('transfer_type', transfer_type)
    experiment.add_tag(dataset_type)
    experiment.log_parameters(hyperparams)
    network_type, hierarch_type = get_network_type(hyperparams)
    experiment.add_tag(network_type)
    experiment.add_tag(hierarch_type)
    
    return experiment
    
def initialize_datasets(user_level_data, subjects_split, hyperparams, hyperparams_features, 
                        validation_set, emotions, liwc_categories, session=None, classes=1):
    liwc_words_for_categories = pickle.load(open(hyperparams_features['liwc_words_cached'], 'rb'))
    if classes!=1 and 'class_weights' in hyperparams:
        class_weights = hyperparams['class_weights']
    else:
        class_weights = None
    data_generator_train = DataGenerator(user_level_data, subjects_split, set_type='train',
                                        hyperparams_features=hyperparams_features,
                                        seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                        posts_per_group=hyperparams['posts_per_group'], post_groups_per_user=hyperparams['post_groups_per_user'],
                                        max_posts_per_user=hyperparams['posts_per_user'], 
                                         compute_liwc=True)
    data_generator_valid = DataGenerator(user_level_data, subjects_split, set_type=validation_set,
                                            hyperparams_features=hyperparams_features,
                                        seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                        posts_per_group=hyperparams['posts_per_group'], 
                                         post_groups_per_user=1,
                                        max_posts_per_user=None, 
                                        shuffle=False,
                                         compute_liwc=True)

    return data_generator_train, data_generator_valid

def initialize_model(hyperparams, hyperparams_features, embedding_matrix, emotions, stopword_list,
                    liwc_categories, logger=None, session=None, transfer=False, classes=1):

    if not logger:
      logger = logging.getLogger('training')
      ch = logging.StreamHandler(sys.stdout)
      # create formatter
      formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
      # add formatter to ch
      ch.setFormatter(formatter)
      # add ch to logger
      logger.addHandler(ch)
      logger.setLevel(logging.DEBUG)
    logger.info("Initializing model...\n")
    # Initialize model
    if hyperparams['hierarchical']:
        model = build_hierarchical_model(hyperparams, hyperparams_features, embedding_matrix, 
                                         emotions, stopword_list, liwc_categories,
                       ignore_layer=hyperparams['ignore_layer'], classes=classes)
    else:
        model = build_model(hyperparams, hyperparams_features, embedding_matrix, 
                                        emotions, stopword_list, liwc_categories,
                       ignore_layer=hyperparams['ignore_layer'], classes=classes)
    if transfer:
        model = build_tl_model(hyperparams, hyperparams_features, embedding_matrix, 
                                        emotions, stopword_list, liwc_categories,
                       ignore_layer=hyperparams['ignore_layer'])
        model.load_weights(hyperparams_features['pretrained_model_path'] + '_weights.h5', by_name=True)
    # Needed just for bert
    if 'bert_layer' not in hyperparams['ignore_layer']:
        initialize_sess(session)                  
    model.summary()
    return model

def train(user_level_data, subjects_split, 
          hyperparams, hyperparams_features, 
          embedding_matrix, emotions, stopword_list, liwc_categories,
          experiment, dataset_type, transfer_type, logger=None,
          validation_set='valid',
          version=0, epochs=50, start_epoch=0,
         session=None, model=None, transfer_layer=False,
         classes=1):
  if not logger:
    logger = logging.getLogger('training')
    ch = logging.StreamHandler(sys.stdout)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    network_type, hierarch_type = get_network_type(hyperparams)
    for feature in ['LIWC', 'emotions', 'numerical_dense_layer', 'sparse_feat_dense_layer', 'user_encoded']:
        if feature in hyperparams['ignore_layer']:
            network_type += "no%s" % feature
    if not transfer_layer:
        model_path='models/%s_%s_%s%d' % (network_type, dataset_type, hierarch_type, version)
    else:
        model_path='models/%s_%s_%s_transfer_%s%d' % (network_type, dataset_type, hierarch_type, transfer_type, version)
        
    
    # Ablate emotions or LIWC
    if 'emotions' in hyperparams['ignore_layer']:
        emotions = []
    if 'LIWC' in hyperparams['ignore_layer']:
        liwc_categories = []

    logger.info("Initializing datasets...\n")
    data_generator_train, data_generator_valid = initialize_datasets(user_level_data, subjects_split, 
                                                                     hyperparams,hyperparams_features,
                                                                     validation_set=validation_set,
                                                                     emotions=emotions, liwc_categories=liwc_categories, 
                                                                    session=session, classes=classes)
    if not model:
        if transfer_layer:
            logger.info("Initializing pretrained model...\n")
        else:
            logger.info("Initializing model...\n")
        model = initialize_model(hyperparams, hyperparams_features, embedding_matrix, 
                                 emotions, stopword_list, liwc_categories, session=session, transfer=transfer_layer,
                                classes=classes)

       
    print(model_path)
    logger.info("Training model...\n")
    model, history = train_model(model, hyperparams,
                                 data_generator_train, data_generator_valid,
                       epochs=epochs, start_epoch=start_epoch,
                      class_weight={0:1, 1:hyperparams['positive_class_weight']} if classes==1 else None,
                      callback_list = [
                          'weights_history',
                          'lr_history',
                          'outputs_history_valid',
                          'outputs_history_train',
                          'reduce_lr_plateau',
                          'lr_schedule'
                                      ],
                      model_path=model_path, workers=4,
                                validation_set=validation_set)
    logger.info("Saving model...\n")
    try:
        save_model_and_params(model, model_path, hyperparams, hyperparams_features)
        experiment.log_parameter("model_path", model_path)
    except:
        logger.error("Could not save model.\n")

    return model, history

