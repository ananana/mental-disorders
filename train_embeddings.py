import csv
import numpy as np
import pandas as pd
import json
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from mittens import Mittens

# Load glove embeddings
embeddings_dim = 100
glove_filepath = '/home/anasab/resources/glove.twitter.27B/glove.twitter.27B.%dd.txt' % embeddings_dim

def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed
print("Loading original glove embeddings...")
pre_glove = glove2dict(glove_filepath)

# Load text data
print("Loading text data...")
#writings_clpsych = pd.DataFrame.from_dict(json.load(open('writings_df_clpsych_all.json')))
#all_texts = "\n".join(writings_clpsych.text.values)
#pickle.dump(all_texts, open("clpsych_text_dump.pkl", "wb+"))
all_texts = pickle.load(open("data/all_texts_clpsych_erisk.pkl", "rb"))
#all_texts = pickle.load(open("texts_erisk_selfharm.pkl", "rb"))

#print("Tokenizing text data...")
#tt = TweetTokenizer()
#all_texts_tokenized = tt.tokenize(all_texts)
#sw = stopwords.words("english")
#all_texts_tokenized_clean = [token.lower() for token in all_texts_tokenized if (token.lower() not in sw)
#                            and re.match("^[a-z]*$", token.lower())]
#pickle.dump(all_texts_tokenized_clean, open("clpsych_text_tokenized.pkl", "wb+"))
#all_texts_tokenized_clean = pickle.load(open("clpsych_text_tokenized.pkl", "rb"))
#oov = [token for token in all_texts_tokenized_clean if token not in pre_glove.keys()]
#def get_rareoov(xdict, val):
#    return [k for (k,v) in Counter(xdict).items() if v<=val]
#def get_freqw(xdict, topn=10000):
#    return [k for (k,v) in Counter(xdict).most_common(topn)]
# oov_rare = get_rareoov(oov, 2)
## corp_vocab = list(pre_glove.keys()) + 
# corp_vocab = list(set(oov) - set(oov_rare))
#corp_vocab = get_freqw(all_texts_tokenized_clean, 10000)
#pickle.dump(corp_vocab, open("vocab_clpsych_10000.pkl", "wb+"))
corp_vocab = pickle.load(open("all_vocab_clpsych_erisk_stop_40000.pkl", "rb"))
original_glove = {k:v for k,v in pre_glove.items() if k in corp_vocab}
pickle.dump(original_glove, open("original_glove_clpsych_erisk_stop_40000.pkl", "wb+"))

# Train with mittens
print("Computing cooccurrence matrix...") 
#cv = CountVectorizer(ngram_range=(1,1), vocabulary=corp_vocab)
#X = cv.fit_transform([all_texts])
#Xc = (X.T * X)
#Xc.setdiag(0)
#coocc_ar = Xc.toarray()
#pickle.dump(coocc_ar, open("coocc_mat_clpsych_erisk_stop_40000.pkl", "wb+"), protocol=4)
coocc_ar = pickle.load(open("coocc_mat_clpsych_erisk_stop_40000.pkl", "rb"))
#coocc_ar = pickle.load(open("coocc_mat_clpsych_oov2.pkl", "rb"))

print("Training with mittens...")
mittens_model = Mittens(n=100, max_iter=1000, mittens=0.2)
new_embeddings = mittens_model.fit(
    coocc_ar,
    vocab=corp_vocab,
    initial_embedding_dict= pre_glove)

print("Serializing embeddings...")
newglove = dict(zip(corp_vocab, new_embeddings))
f = open("finetuned_glove_clpsych_erisk_stop_40000.pkl","wb")
pickle.dump(newglove, f)
f.close()
