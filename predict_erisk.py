from load_save_model import load_params, load_saved_model_weights
from resource_loading import load_stopwords
import json
from EriskDataGenerator import EriskDataGenerator

def predict(run_nr):
    model_paths = {
        1: 'models/lstm_selfharm_hierarchical107',    # 80 posts per chunk, trained on self-harm
        2: 'models/lstm_selfharm_hierarchical113',    # 10 posts per chunk, trained on self-harm
        3: 'models/lstm_selfharm_hierarchical113',
        # 4: 'models/lstm_selfharm_hierarchical107'     # 10 posts per chunk, pre-trained on eRisk
                                                      # depression+anorexia, trained on eRisk self-harm
    }
    model_path = model_paths[run_nr]
    hyperparams, hyperparams_features = load_params(model_path)
    with open('config.json') as f:
        config = json.load(f)
    stopwords_list = load_stopwords(config['stopwords_list'])
    model = load_saved_model_weights(model_path, hyperparams, config, 
                                                      stopwords_dim=len(stopwords_list),
                                                      h5=True)

    data_generator = EriskDataGenerator()

    # Reading eRisk data
    data_round1 = {
    "redditor": 338, "content": "", 
    "date": "2014-12-12T04:21:13.000+0000", 
    "id": 168996, 
    "title": "    Copy the Reindeer", 
    "number": 1, 
    "nick": "subject8081"}
    {"redditor": 339, 
    "content": "    When I don't have the aisle seat and have to climb over people to use the bathroom. I have a tiny girl bladder.", 
    "date": "2013-10-10T13:17:01.000+0000", 
    "id": 169297, 
    "title": "", 
    "number": 1, 
    "nick": "subject2621"}
    {"redditor": 340, 
    "content": "    I have a question about being a visitor in Nioh(Random encounters)", 
    "date": "2017-05-09T17:01:50.000+0000", 
    "id": 169531, "title": "    Nioh - Become a visitor", 
    "number": 1, 
    "nick": "subject992"}
    
    data_round2 = {
    "redditor": 340, 
    "content": "    New text", 
    "date": "2017-05-09T17:02:50.000+0000", 
    "id": 169532, 
    "title": "    Nioh - Become a visitor", "number": 2, "nick": "subject992"}
    data_generator.add_data_round(data_round1)
    data_generator.add_data_round(data_round2)

    predictions = model.predict(generator)

    # TODO: add rolling average
    # TODO: emit zeros if number of datapoints is too small
    return predictions