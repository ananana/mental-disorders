This directory contains the code needed to load UPV's models for Task 2 (HAN) and use them for inference (or training) on eRisk data.

## Install
The required libraries are:
- tensorflow2, keras
- numpy, nltk, sklearn, pandas

The full list of packages and versions I used is found in `requirements.txt` (may contain some unnecessary ones)

The required data:
- expects a `config.json` in the root directory (containing paths to resource files)
- expects files with model weights and their hyperparameter configurations in `models/`
- expects external resources (NRC lexicon, LIWC lexicon, pre-trained embeddings) in `resources/`
- expects vocabulary files and a LIWC cache file in `data/`

They can be downloaded from the analogous subdirectories in the Azure Storage, under `upv-models`.

## Usage

`predict_erisk.py` illustrates how models can be loaded and used to generate predictions on eRisk data.

`predict(run_nr, data_rounds)` can be used to obtain predictions (scores and alerts for each user, for each datapoint) from a specific trained model given some data obtained from the eRisk server, across one or more rounds of interaction with the server.

For more control, `EriskDataGenerator()` (along with `model.predict_step()`) can be used directly, which allows to incrementally add new datapoints to the generator as they are received from the server (without creating a new loader for each new datapoint).