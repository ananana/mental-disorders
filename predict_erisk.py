from load_save_model import load_params, load_saved_model_weights
from resource_loading import load_stopwords
import json
from EriskDataGenerator import EriskDataGenerator
import numpy as np

RUNS_MODEL_PATHS = {
        1: 'models/lstm_selfharm_hierarchical107',    # 80 posts per chunk, trained on self-harm
        2: 'models/lstm_selfharm_hierarchical113',    # 10 posts per chunk, trained on self-harm
        3: 'models/lstm_selfharm_hierarchical113',    # 10 posts per chunk, trained on self-harm, rolling average predictions
        # 4: 'models/lstm_selfharm_hierarchical107'     # 10 posts per chunk, pre-trained on eRisk
                                                      # depression+anorexia, trained on eRisk self-harm
    }

def predict(run_nr, data_rounds, alert_threshold=0.5, rolling_window=50, conservative_alerts=True):
    """
    Expects a run_nr corresponding to the solution to be used for generating predictions.
    Solutions correspond to the ones described in the PDF document - more details on their
    implementation and performance are found there.
    Every solution uses a different trained model, or a different strategy for generating
    predictions.
    
    Parameters:
    run_nr: integer representing the solution/run to be used
    data_rounds: a list of dictionaries containing data corresponding to one post / one round
                in the stream, in the format used by the eRisk server
    
    Returns:
    a list of predictions (0/1)
    """
    
    model_path = RUNS_MODEL_PATHS[run_nr]
    hyperparams, hyperparams_features = load_params(model_path)

    model = load_saved_model_weights(model_path, hyperparams, hyperparams_features, 
                                                      h5=True)

    data_generator = EriskDataGenerator(hyperparams_features=hyperparams_features,
                                seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                     max_posts_per_user=None,
                                    posts_per_group=hyperparams['posts_per_group'],
                                    post_groups_per_user=None, 
                                    shuffle=False,
                                            compute_liwc=True)

    for data_round in data_rounds:
        data_generator.add_data_round(data_round)

    predictions = model.predict(data_generator)

    # TODO: implement rolling average per user
    # Use rolling average on prediction scores
    if run_nr==3:
        rolling_predictions = []
        # The first predictions will be copied
        rolling_predictions[:rolling_window-1] = predictions[:rolling_window-1]
        # rolling average over predictions
        rolling_predictions.extend(np.convolve(predictions, np.ones(rolling_window), 'valid') / rolling_window)
        # predictions = rolling_predictions

    if conservative_alerts and len(data_rounds) < hyperparams['posts_per_group']:
        alerts = [0 for p in predictions]
    else:
        alerts = [int(p >= 0.5) for p in predictions]

    return alerts

if __name__=='__main__':
        # Reading eRisk data
    data_round1 = [{
    "redditor": 338, "content": "", 
    "date": "2014-12-12T04:21:13.000+0000", 
    "id": 168996, 
    "title": "    Copy the Reindeer", 
    "number": 1, 
    "nick": "subject8081"},
    {"redditor": 339, 
    "content": "    When I don't have the aisle seat and have to climb over people to use the bathroom. I have a tiny girl bladder.", 
    "date": "2013-10-10T13:17:01.000+0000", 
    "id": 169297, 
    "title": "", 
    "number": 1, 
    "nick": "subject2621"},
    {"redditor": 340, 
    "content": "    I have a question about being a visitor in Nioh(Random encounters)", 
    "date": "2017-05-09T17:01:50.000+0000", 
    "id": 169531, "title": "    Nioh - Become a visitor", 
    "number": 1, 
    "nick": "subject992"}]
    
    data_round2 = [{
    "redditor": 340, 
    "content": "    New text", 
    "date": "2017-05-09T17:02:50.000+0000", 
    "id": 169532, 
    "title": "    Nioh - Become a visitor", "number": 2, "nick": "subject992"}]

    predict(run_nr=1, data_rounds=[data_round1, data_round2])