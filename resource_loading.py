from liwc_readDict import readDict

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
    for (w, c) in readDict():
        if c not in liwc_dict:
            liwc_dict[c] = []
        liwc_dict[c].append(w)
    return liwc_dict