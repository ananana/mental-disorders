class WeightsHistory(callbacks.Callback):
    def __init__(self, logs={}):
        super(WeightsHistory, self).__init__()
    def on_train_begin(self, logs={}):
        self.log_weights(0)
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            self.log_weights(epoch)
    def log_weights(self, step):
        for layer in self.model.layers:
            try:
                experiment.log_histogram_3d(layer.get_weights()[0], 
                                            name=layer.name + "_weight", step=step)
            except Exception as e:
#                 logger.debug("Logging weights error: " + layer.name + "; " + str(e) + "\n")
                # Layer probably does not exist
                pass

class OutputsHistory(callbacks.Callback):
    def __init__(self, logs={}, generator=None, generator_type=""):
        super(OutputsHistory, self).__init__()
        self.generator_type = generator_type
        if generator:
            self.generator = generator
        elif generator_type:
            self.generator = DataGenerator(user_level_data, subjects_split, 
                                     set_type=generator_type, 
                                   hierarchical=hyperparams['hierarchical'],
                                seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                     max_posts_per_user=None,
                                   pad_with_duplication=False,
                                    posts_per_group=hyperparams['posts_per_group'],
                                    post_groups_per_user=None, 
                                           liwc_words_for_categories=liwc_words_for_categories,
                                           compute_liwc=True,
                                         emotions=emotions, liwc_categories=liwc_categories,
                                     sample_seqs=False, shuffle=False)
    def on_train_begin(self, logs={}):
        self.log_outputs(0)
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 2 == 0:
            self.log_outputs(epoch)
    def log_outputs(self, step):
        try:
            experiment.log_histogram_3d(self.model.predict(self.generator,  verbose=1, steps=2),
                                        name='output_%s' % self.generator_type, step=step)
        except Exception as e:
            logger.debug("Logging outputs error: " + str(e) + "\n")
#                 Layer probably does not exist
            pass

class ActivationsAttention(callbacks.Callback):
    def __init__(self, logs={}, generator=None, generator_type=""):
        super(ActivationsAttention, self).__init__()
        self.generator_type = generator_type
        if generator:
            self.generator = generator
        elif generator_type:
            self.generator = DataGenerator(user_level_data, subjects_split, 
                                     set_type=generator_type, 
                                   hierarchical=hyperparams['hierarchical'],
                                seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                     max_posts_per_user=None,
                                   pad_with_duplication=False,
                                    posts_per_group=hyperparams['posts_per_group'],
                                    post_groups_per_user=None, 
                                     sample_seqs=False, shuffle=False)
    def on_train_begin(self, logs={}):
        self.log_outputs(0)
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            self.log_outputs(epoch)
    def log_outputs(self, step):
        try:
            experiment.log_histogram_3d(self.model.get_layer('attention_user').output.eval())
        except Exception as e:
            logger.debug("Logging activations error: " + str(e) + "\n")
            pass

class LRHistory(callbacks.Callback):
    def __init__(self, logs={}):
        super(LRHistory, self).__init__()
    
    def on_epoch_begin(self, epoch, logs={}):
        self.log_lr()
        
    def log_lr(self):
        lr = K.eval(self.model.optimizer.lr)
        logger.debug("Learning rate is %f...\n" % lr)
        experiment.log_parameter('lr', lr)

class FreezeLayer(callbacks.Callback):
    def __init__(self, logs={}, patience=5, layer={'user_encoder':'embeddings_layer'}, verbose=1, set_to=False):
        super(FreezeLayer, self).__init__()
        self.freeze_epoch = patience
        self.freeze_layer = layer
        self.verbose = verbose
        self.set_to = set_to

    def on_epoch_begin(self, epoch, logs={}):
        if type(self.freeze_layer)==dict:
            submodel = self.model.get_layer(list(self.freeze_layer.keys())[0])
        else:
            submodel = self.model
        logging.debug("Trainable embeddings", submodel.get_layer(self.freeze_layer).trainable)
        if epoch == self.freeze_epoch:
            try:
                layer = submodel.get_layer(self.freeze_layer)
                old_value = layer.trainable
                layer.trainable = self.set_to
                # TODO: does this reset the optimizer? should I also compile the top-level model?
                self.model.compile(hyperparams['optimizer'], binary_crossentropy_custom,
                  metrics=[metrics_class.f1_m, metrics_class.precision_m, metrics_class.recall_m])
                if self.verbose:
                    logging.debug("Setting %s layer from %s to trainable=%s...\n" % (layer.name, old_value,
                                                                   submodel.get_layer(self.freeze_layer).trainable))
            except Exception as e:
                # layer probably does not exist
                pass