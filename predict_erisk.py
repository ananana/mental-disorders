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

def scores_to_alerts(predictions_dict, conservative_alerts=False, 
                     alert_threshold=0.5, rolling_window=0):
    '''Generates alerts decisions (1/0) from a dictionary of prediction scores per user
    Parameters:
        predictions_dict: dictionary with ordered predictions per user (indexed by user id) 
        rolling_window: window of rolling average to be computed across prediction scores
            history for a given user in order to get a "smoothed" prediction for each datapoint
            If 0, then no rolling average is computed.
        conservative_alerts: if True, will only emit positive alerts if enough input posts are
            used for prediction (will only trust predictions based on at least as many posts
            as were used in one datapoint in the training stage)
        posts_per_datapoint: integer denoting number of posts per datapoint used in the training stage
            used in case of conservative_alerts=True
        alert_threshold: threshold on the score value above which to emit a positive alert
        Returns: nested dictionary indexed by users, including the original prediction score
            ('scores' key) and the alert value (1/0) (the 'alerts' key)'''
    users = predictions_dict.keys()
    scores_per_user = dict(predictions_dict)
    def _rolling_average(scores, window):
        if window < len(scores):
            return scores
        rolling_predictions = []
        rolling_predictions[:rolling_window-1] = scores[:rolling_window-1]
        rolling_predictions.extend(np.convolve(scores, np.ones(rolling_window), 'valid') / rolling_window)
        return rolling_predictions
    if rolling_window:
        scores_per_user = {u: _rolling_average(scores_per_user[u], rolling_window) for u in users}
    alerts_per_user = {}
    for u in users:
        if conservative_alerts:
            alerts_per_user[u] = [0 for p in scores_per_user[u]]
        else:
            alerts_per_user[u] = [int(p>=alert_threshold) for p in scores_per_user[u]]
    return {u: {'scores': scores_per_user[u], 'alerts': alerts_per_user[u]} for u in users}

def predict(run_nr, data_rounds, alert_threshold=0.5, rolling_window=50, conservative_alerts=True):
    """
    Expects a run_nr corresponding to the solution to be used for generating predictions.
    Solutions correspond to the ones described in the PDF document - more details on their
    implementation and performance are found there.
    Every solution uses a different trained model, or a different strategy for generating
    predictions.
    Will generate one for prediction every x posts for each user, where x is the number of posts 
    in one chunk/datapoint used in training for this model
    If number of posts available < x, it will still generate a prediction, alerts will be 0
    by default (in the conservative setting, controlled by a flag)
    Parameters:
    run_nr: integer representing the solution/run to be used
    data_rounds: a list of dictionaries containing data corresponding to one post / one round
                in the stream, in the format used by the eRisk server
    
    Returns:
    a dictionary of scores and alerts (1/0) per user in the input data
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


    predictions_per_user = {}
    for dp in data_generator:
        prediction = model.predict_step(dp)
        u = dp[1][0]
        if u not in predictions_per_user:
            predictions_per_user[u] = []
        predictions_per_user[u].append(prediction.numpy()[0].item())
    alerts_per_user = scores_to_alerts(predictions_per_user, rolling_window=rolling_window,
        alert_threshold=alert_threshold,
        conservative_alerts=(conservative_alerts and len(data_rounds) < hyperparams['posts_per_group']))
    return alerts_per_user

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


    print(predict(run_nr=1, data_rounds=[data_round1, data_round2]))
