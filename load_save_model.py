def save_model_and_params(model, model_path, hyperparams, hyperparams_features):
    model.save_weights(model_path + "_weights.h5", save_format='h5')
#     model.save(model_path + "_model.model")
#     model.save(model_path + "_model.h5")
    with open(model_path + '.hp.json', 'w+') as hpf:
        hpf.write(json.dumps({k:v for (k,v) in hyperparams.items() if k!='optimizer'}))
    with open(model_path + '.hpf.json', 'w+') as hpff:
        hpff.write(json.dumps(hyperparams_features))

def load_params(model_path):
    with open(model_path + '.hp.json', 'r') as hpf:
        hyperparams = json.loads(hpf.read())
    with open(model_path + '.hpf.json', 'r') as hpff:
        hyperparams_features = json.loads(hpff.read())
    return hyperparams, hyperparams_features

def load_saved_model(model_path, hyperparams):
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    dependencies = {
    'f1_m': metrics_class.f1_m,
    'auc': metrics_class.auc,
    'precision_m': metrics_class.precision_m,
    'recall_m': metrics_class.recall_m,
    'binary_crossentropy_custom': binary_crossentropy_custom,
    'BertLayer': BertLayer
    }
    loaded_model = load_model(model_path + "_model.h5", custom_objects=dependencies)
#     loaded_model = load_model(model_path + "_model.model", custom_objects=dependencies)
    return loaded_model

def load_saved_model_weights(model_path, hyperparams, emotions, stopword_list, liwc_categories, classes, h5=False):
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    dependencies = {
    'f1_m': metrics_class.f1_m,
    'auc': metrics_class.auc,
    'precision_m': metrics_class.precision_m,
    'recall_m': metrics_class.recall_m,
    'binary_crossentropy_custom': binary_crossentropy_custom,
    'BertLayer': BertLayer
    }
    loaded_model = initialize_model(hyperparams=hyperparams, hyperparams_features=hyperparams_features, 
                                    embedding_matrix=embedding_matrix, 
                                 emotions=emotions, stopword_list=stopword_list, liwc_categories=liwc_categories,
                                classes=classes)
    loaded_model.summary()
#     loaded_model.load_weights(model_path + "_weights")
    path = model_path + "_weights"
    by_name = False
    if h5:
        path += ".h5"
        by_name=True
    loaded_model.load_weights(path, by_name=by_name)
    #     loaded_model = load_model(model_path + "_model.model", custom_objects=dependencies)
    return loaded_model