import datasets    
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

_DESCRIPTION_T12 = """\
Metrics for measuring the performance of prediction models for eRisk 2021 Task 2 and 3.
Include decision-based performance metrics: decision-based F1, lantency-weighted F1, ERDE score.
"""

_CITATION = ""

_KWARGS_DESCRIPTION_T12 = """
Calculates how good are predictions given some references, using certain scores.
Predictions and references are expected to be ordered chronologically (in order of
their appearance in the input stream).
Decision-based metrics consider predictions at the user level, where a decision for a
given user is considered positive if any prediction for that user is positive.
Args:
    predictions: list of predictions to score. Each prediction
        should be an integer in {0, 1}.
    references: list of references for each prediction. Each
        reference should be a dictionary with keys 'label' and 'user',
        containing the true label for the given example (integer in {0, 1}),
        and the user who authored the given datapoint (as a string).
        Labels are expected to be consistent for the same user across references.
    posts_per_datapoint: the number of user posts that are used to generate one prediction
        and correspond to one label in the input.
Returns:
    precision: decision-based precision
    recall: decision-based recall
    f1: decision-based f
    latency_f1: f1 weighted by the median latency for positive alerts
    erde: error score penalizing late alerts
Examples:

    >>> erisk_metric = EriskScoresT1T2()
    >>> results = erisk_metric.compute(
        references=[{'user': 'subject14', 'label': 0},
        {'user': 'subject15', label: 1}], predictions=[0, 1],
        posts_per_datapoint=50)
    >>> print(results)
    {'f1': 1.0,
    'f1_latency': 1.0,
    'precision': 1.0,
    'recall': 1.0,
    'erde5': 0.0,
    'erde50': 0.0}
"""
    
def _penalty(k, p=0.0078):
    return -1 + 2 / (1 + np.exp(-p * (k - 1)))

def _lc(k, o):
    return 1 - (1 / (1 + np.exp(k - o)))



class EriskScoresT1T2(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION_T12,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION_T12,
            features=datasets.Features({
                'predictions': datasets.Value('int64'),
                'references': datasets.DatasetDict({'label': datasets.Value('int64'),
                               'user': datasets.Value('string')
                              })
            }),
            codebase_urls=[],
            reference_urls=[],
        )
    
    def _latency(self, predictions, references, posts_per_datapoint):
        assert(len(predictions)==len(references)), \
               "Number of predictions not equal to number of references: %d vs %d." % (len(predictions), len(references))

        predictions_per_user = {}
        labels_per_user = {}
        for i in range(len(predictions)):
            u = references[i]['user']
            l = references[i]['label']
            p = predictions[i]
            if u not in predictions_per_user:
                predictions_per_user[u] = []
            predictions_per_user[u].append(p)
            if u in labels_per_user:
                assert(labels_per_user[u] == l), "Inconsistent labels for same user: %s" % u
            else:
                labels_per_user[u] = l
        users = list(labels_per_user.keys())
        latencies = []
        for u in users:
            # Latency only relevant for true positives
            if labels_per_user[u] != 1 or sum(predictions_per_user[u]) == 0:
                continue
            i = 0
            p = predictions_per_user[u][i]
            # Minimum latency has to be the number of posts used for the first prediction,
            # assuming we predict 0s by default (before the model generated any predictions)
            latency = posts_per_datapoint
            while (p != 1) and (i < len(predictions_per_user[u])):
                latency += posts_per_datapoint
                p = predictions_per_user[u][i]
                i += 1
                
            latencies.append(latency)
        median_penalty = _penalty(np.median(latencies))
        print(latencies, median_penalty)
        return median_penalty
    
    def _erde(self, predictions, references, posts_per_datapoint, o):
        assert(len(predictions)==len(references)), \
               "Number of predictions not equal to number of references: %d vs %d." % (len(predictions), len(references))

        predictions_per_user = {}
        labels_per_user = {}
        for i in range(len(predictions)):
            u = references[i]['user']
            l = references[i]['label']
            p = predictions[i]
            if u not in predictions_per_user:
                predictions_per_user[u] = []
            predictions_per_user[u].append(p)
            if u in labels_per_user:
                assert(labels_per_user[u] == l), "Inconsistent labels for same user: %s" % u
            else:
                labels_per_user[u] = l
        users = list(labels_per_user.keys())
        penalties = []
        for u in users:
            # Latency only relevant for true positives
            if labels_per_user[u] != 1 or sum(predictions_per_user[u]) == 0:
                continue
            i = 0
            p = predictions_per_user[u][i]
            latency = posts_per_datapoint
            while (p != 1) and (i < len(predictions_per_user[u])):
                latency += posts_per_datapoint
                p = predictions_per_user[u][i]
                i += 1
                
            penalties.append(latency)
        erde = np.median([_lc(p, o) for p in penalties])
        return erde
        

    def _compute(self, predictions, references, posts_per_datapoint):
        assert(len(predictions)==len(references)), \
               "Number of predictions not equal to number of references: %d vs %d." % (len(predictions), len(references))
        predictions_per_user = {}
        labels_per_user = {}
        for i in range(len(predictions)):
            u = references[i]['user']
            l = references[i]['label']
            p = predictions[i]
            if u not in predictions_per_user:
                predictions_per_user[u] = p
            # User-level prediction is 1 if any 1 was emitted, otherwise it's 0
            predictions_per_user[u] = (p or predictions_per_user[u])
            if u in labels_per_user:
                assert(labels_per_user[u] == l), "Inconsistent labels for same user: %s" % u
            else:
                labels_per_user[u] = l
        users = list(labels_per_user.keys())
        y_true = [labels_per_user[u] for u in users]
        y_pred = [predictions_per_user[u] for u in users]
        penalty_score = self._latency(predictions, references, posts_per_datapoint)
        return {"precision": precision_score(y_true, y_pred),
               "recall": recall_score(y_true, y_pred),
               "f1": f1_score(y_true, y_pred),
               "latency_f1": f1_score(y_true, y_pred) * (1 - penalty_score),
               "erde5": self._erde(predictions, references, posts_per_datapoint, 5),
               "erde50": self._erde(predictions, references, posts_per_datapoint, 50)}