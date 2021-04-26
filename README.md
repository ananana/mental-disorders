This repository contains code for training and using deep learning models for mental disorder detection from social media data.

## Install
The required libraries are:
- tensorflow2, keras
- numpy, nltk, sklearn, pandas

The full list of packages and versions I used is found in `requirements.txt` (may contain some unnecessary ones)

The required data:
- expects a `config.json` in the root directory (containing paths to resource files)
- for generating predictions, expects files with trained model weights and their hyperparameter configurations in `models/`
- expects external resources (NRC lexicon, LIWC lexicon, pre-trained embeddings) in `resources/`
- expects vocabulary files and a LIWC cache file in `data/`


## Usage

`predict_erisk.py` illustrates how models can be loaded and used to generate predictions on eRisk data.

`predict(run_nr, data_rounds)` can be used to obtain predictions (scores and alerts ("decision"s) for each user ("nick"), for each datapoint) from a specific trained model given some data obtained from the eRisk server, across one or more rounds of interaction with the server.

For more control, `EriskDataGenerator()` (along with `model.predict_step()`) can be used directly, which allows to incrementally add new datapoints to the generator as they are received from the server (without creating a new loader for each new datapoint).
