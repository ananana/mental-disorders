def encode_emotions(tokens, emotion_lexicon, emotions, relative=True):
    text_len = len(tokens)
    encoded_emotions = [0 for e in emotions]
    for i, emotion in enumerate(emotions):
        try:
            emotion_words = [t for t in tokens if t in emotion_lexicon[emotion]]
            if relative and len(tokens):
                encoded_emotions[i] = len(emotion_words) / len(tokens)
            else:
                encoded_emotions[i] = len(emotion_words)
        except ValueError:
            print("Emotion not found.")
    return encoded_emotions



def tokenize_fields(writings_df, tokenize_fct, columns=['title', 'text']):
    for c in columns:
        writings_df['tokenized_%s' % c] = writings_df['%s' % c].apply(lambda t: tokenize_fct(t) 
                                                                if type(t)==str and t else None)
        writings_df['%s_len' % c] = writings_df['tokenized_%s' % c].apply(lambda t: len(t) 
                                                                    if type(t)==list and t else None)
    return writings_df

def encode_pronouns(tokens, pronouns={"i", "me", "my", "mine", "myself"}, relative=True):
    if not tokens:
        return 0
    text_len = len(tokens)
    nr_pronouns = len([t for t in tokens if t in pronouns])
    if relative and text_len:
        return nr_pronouns/text_len
    else:
        return nr_pronouns

def encode_stopwords(tokens, stopwords):
    encoded_stopwords = [0 for s in stopword_list]
    if not tokens:
        return encoded_stopwords
    for i, stopword in enumerate(stopwords):
        if stopword in tokens:
            encoded_stopwords[i] += 1
    return encoded_stopwords