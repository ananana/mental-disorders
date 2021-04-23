# importing the requests library
import requests
import json

# api-endpoint
URL_GET = "https://erisk.irlab.org/challenge-t1/getwritings/%s"
URL_POST = "https://erisk.irlab.org/challenge-t/submit/%s/%d"

TOKEN = ""

def get_users():

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {}

    # sending get request and saving the response as response object
    r = requests.get(url=URL_GET%TOKEN, params=PARAMS)

    # extracting data in json format
    data = r.json()

    return data

def process_data(json_data):
    subjects = [d['nick'] for d in json_data]
    return subjects

def get_subjects(filepath='data.jl'):
    subjects = []
    with open(filepath) as f:
        for line in f:
            subjects.append(json.loads(line)['nick'])
    return subjects

def send_prediction(run_nr, predictions):
    data = []
    for subject, score, label in predictions:
        data.append({
            'nick': subject,
            'decision': label,
            'score': score
        })
    print('prediction len', len(data))
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response = requests.post(url=URL_POST%(TOKEN,run_nr), data=json.dumps(data), headers=headers)

    return response

def get_predictions_dummy(data, predictions, scores):
    data = process_data(data)
    subjects = get_subjects()
    # dummy predictions
    predictions = [(s, predictions[s], scores[s]) for s in subjects]
    return predictions

def get_response_from_file(model, run, rnd):
   predictions_json = read_data('response_%s_run%d_rnd%d.json')
   return predictions_json

def get_predictions(data):
    data = process_data(data)
    subjects = get_subjects()
    # dummy predictions
    predictions = [(s, 0, 0.6) for s in subjects]
    return predictions

def serialize_data(data):
    round = data[0]['number']
    with open('data%d.jl'%round, 'w+') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    with open('subjects%d.txt'%round, 'w+') as f:
        for item in process_data(data):
            f.write(item + '\n')

def read_data(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__=='__main__':
    data = get_users()

    print('len data', len(data), data)
    serialize_data(data)


    for run in range(5):

        print('run', run)
        predictions = get_predictions(data)
        print(send_prediction(run, predictions))

        # You get new round once you submit your results for run 5.
        # What if you submit for run 5 in the beginning tho? Still doesn't give you the new ones right after run 5. It waits to get all of them that's it.
        # You should verify when building data for a new round that you are on the right round... that's it.
        # But it seems to be working properly.

        print(get_users())
