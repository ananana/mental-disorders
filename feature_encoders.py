from nltk.corpus import stopwords
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

def encode_stopwords(tokens, stopwords_list=None):
    if not stopwords_list:
        stopwords_list = stopwords.words("english") 
    encoded_stopwords = [0 for s in stopwords_list]
    if not tokens:
        return encoded_stopwords
    for i, stopword in enumerate(stopwords_list):
        if stopword in tokens:
            encoded_stopwords[i] += 1
    return encoded_stopwords

    
def encode_liwc_categories_full(tokens, liwc_categories, liwc_words_for_categories, relative=True):
    categories_cnt = [0 for c in liwc_categories]
    if not tokens:
        return categories_cnt
    text_len = len(tokens)
    for i, category in enumerate(liwc_categories):
        category_words = self.liwc_dict[category]
        for t in tokens:
            for word in category_words:
                if t==word or (word[-1]=='*' and t.startswith(word[:-1])) \
                or (t==word.split("'")[0]):
                    categories_cnt[i] += 1
                    break # one token cannot belong to more than one word in the category
        if relative and text_len:
            categories_cnt[i] = categories_cnt[i]/text_len
    return categories_cnt
    
    
def encode_liwc_categories(tokens, liwc_categories, liwc_words_for_categories, relative=True):
    categories_cnt = [0 for c in liwc_categories]
    if not tokens:
        return categories_cnt
    text_len = len(tokens)
    for i, category in enumerate(liwc_categories):
        for t in tokens:
            if t in liwc_words_for_categories[category]:
                categories_cnt[i] += 1
        if relative and text_len:
            categories_cnt[i] = categories_cnt[i]/text_len
    return categories_cnt