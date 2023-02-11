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

## Model

![architecture_full](https://user-images.githubusercontent.com/1269090/117694579-99ac2e00-b1bf-11eb-8cc0-0ba6c79272c1.png)


## Publications

The code in this repository has been used for experiments published in several papers. If using this resource, please cite the relevant papers:

- _Explainability of Depression Detection on Social Media: From Deep Learning Models to Psychological Interpretations and Multimodality_, Ana-Sabina Uban, Berta Chulvi, Paolo Rosso, in Early Detection of Mental Health Disorders by Social Media Monitoring, 2022
- _Detecting early signs of depression in the conversational domain: The role of transfer learning in low-resource scenarios_, Petr Lorenc, Ana-Sabina Uban, Paolo Rosso, Jan Šedivý, NLDB 2022
- _Multi-aspect transfer learning for detecting low resource mental disorders on social media_, Ana Sabina Uban, Berta Chulvi, Paolo Rosso, LREC 2022
- _On the Explainability of Automatic Predictions of Mental Disorders from Social Media Data_, Ana Sabina Uban, Berta Chulvi, Paolo Rosso, NLDB 2021
- _An emotion and cognitive based analysis of mental health disorders from social media data_, Ana-Sabina Uban, Berta Chulvi, Paolo Rosso, Future Generation Computer Systems, 2021
- _Upv-symanto at erisk 2021: Mental health author profiling for early risk prediction on the internet_, Angelo Basile, Mara Chinea-Rios, Ana-Sabina Uban, Thomas Müller, Luise Rössler, Seren Yenikent, María Alberta Chulvi-Ferriols, Paolo Rosso, Marc Franco-Salvador, CLEF 2021
- _Deep learning architectures and strategies for early detection of self-harm and depression level prediction_, Ana-Sabina Uban, Paolo Rosso, CLEF 2020


