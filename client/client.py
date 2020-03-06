# importing the requests library
import requests
import json

# api-endpoint
URL_GET = "https://erisk.irlab.org/challenge-service/getwritings/%s"
URL_POST = "https://erisk.irlab.org/challenge-service/submit/%s/%d"
TOKEN = "q95QLQYOeqpMuwrRdggKZ1F614619WSGS9TEyQl2bZ4"

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

def send_prediction(run_nr, predictions):
    data = []
    for subject, score, label in predictions:
        data.append({
            'nick': subject,
            'decision': label,
            'score': score
        })
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    response = requests.post(url=URL_POST%(TOKEN,run_nr), data=json.dumps(data), headers=headers)

    return response

def get_predictions(data):
    subjects = process_data(data)
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

if __name__=='__main__':
    data = get_users()
    print(data)
    serialize_data(data)

    for run in [1, 0, 2, 4, 3]:

        print(run)
        predictions = get_predictions(get_users())
        print(send_prediction(run, predictions))
        # You get new round once you submit your results for run 5.
        # What if you submit for run 5 in the beginning tho? Still doesn't give you the new ones right after run 5. It waits to get all of them that's it.
        # You should verify when building data for a new round that you are on the right round... that's it.
        # But it seems to be working properly.
        print(get_users())
