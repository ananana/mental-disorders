from liwc_readDict import readDict
import pickle
import numpy as np

def load_NRC(nrc_path):
    word_emotions = {}
    emotion_words = {}
    with open(nrc_path) as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            word, emotion, label = line.split()
            if word not in word_emotions:
                word_emotions[word] = set()
            if emotion not in emotion_words:
                emotion_words[emotion] = set()
            label = int(label)
            if label:
                word_emotions[word].add(emotion)
                emotion_words[emotion].add(word)
    return emotion_words

def load_LIWC(path):
    liwc_dict = {}
    for (w, c) in readDict(path):
        if c not in liwc_dict:
            liwc_dict[c] = []
        liwc_dict[c].append(w)
    return liwc_dict

def load_vocabulary(path):
    vocabulary_list = pickle.load(open(path, 'rb'))
    vocabulary_dict={}
    for i,w in enumerate(vocabulary_list):
        vocabulary_dict[w] = i
    return vocabulary_dict

def load_embeddings(path, embedding_dim, vocabulary_path, voc_size):
    # random matrix with mean value = 0
    voc = load_vocabulary(vocabulary_path)
    embedding_matrix = np.random.random((len(voc)+2, embedding_dim)) - 0.5 # voc + unk + pad value(0)
    cnt_inv = 0
    f = open(path, encoding='utf8')
    for i, line in enumerate(f):
#         print(i)
        values = line.split()
        word = ''.join(values[:-embedding_dim])
        coefs = np.asarray(values[-embedding_dim:], dtype='float32')
        word_i = voc.get(word)
        if word_i is not None:
            embedding_matrix[word_i] = coefs
            cnt_inv += 1
    f.close()

    print('Total %s word vectors.' % len(embedding_matrix))
    print('Words not found in embedding space %d' % (len(embedding_matrix)-cnt_inv))
 
    return embedding_matrix