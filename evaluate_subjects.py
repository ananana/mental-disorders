from metrics_decision_based import EriskScoresT1T2
from DataGenerator import DataGenerator

def evaluate_for_subjects(model, subjects, user_level_data, hyperparams, hyperparams_features,
        alert_threshold=0.5):
    erisk_metricst2 = EriskScoresT1T2()
    threshold = alert_threshold
    for subject in set(subjects):

        try:
            user_level_data_subject = {subject: user_level_data[subject]}
        except:
            continue
        true_label = user_level_data_subject[subject]['label']
       
        print(subject, "Label", true_label)
        predictions = model.predict(DataGenerator(user_level_data_subject, {'test':[subject]}, 
                                             set_type='test', hyperparams_features=hyperparams_features,
                                        seq_len=hyperparams['maxlen'],   
                                            batch_size=hyperparams['batch_size'], # on all data at once
                                             max_posts_per_user=None,
                                            posts_per_group=hyperparams['posts_per_group'],
                                            post_groups_per_user=None,  compute_liwc=True,
                                             shuffle=False), verbose=1)
        for prediction in predictions:
            model_prediction = int(prediction[0]>=threshold)

            print("Prediction: ", prediction, model_prediction)

            erisk_metricst2.add(prediction=model_prediction, reference = {'label': true_label, 'user': subject})
            print('prediction and reference', model_prediction, {'label': true_label, 'user': subject})
    return erisk_metricst2.compute(posts_per_datapoint=hyperparams['posts_per_group'])