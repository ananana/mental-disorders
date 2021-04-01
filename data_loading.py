from collections import Counter
import numpy as np
def load_erisk_data(writings_df, voc_size, emotion_lexicon, emotions,
                    liwc_categories, by_subset=True,
                    pronouns = ["i", "me", "my", "mine", "myself"],
                    train_prop=0.7, valid_prop=0.3, test_slice=2,
                    nr_slices=5,
                    min_post_len=3, min_word_len=1, 
                    user_level=True, vocabulary=None, labelcol='label', label_index=None,
                   logger=None):
#     logger.debug("Loading data...\n")
    
    ## Build vocabulary
    vocabulary_all = {}
    word_freqs = Counter()
    
    for words in writings_df.tokenized_text:
        word_freqs.update(words)
    for words in writings_df.tokenized_title:
        word_freqs.update(words)
    i = 1
   
    training_subjects = list(set(writings_df[writings_df['subset']=='train'].subject))
    test_subjects = list(set(writings_df[writings_df['subset']=='test'].subject))
    
    training_subjects = sorted(training_subjects) # ensuring reproducibility
    valid_subjects_size = int(len(training_subjects) * valid_prop)
    valid_subjects = training_subjects[:valid_subjects_size]
    training_subjects = training_subjects[valid_subjects_size:]
    categories = [c for c in liwc_categories if c in writings_df.columns]
#     logger.debug("%d training users, %d validation users, %d test users." % (
#         len(training_subjects), 
#           len(valid_subjects),
#           len(test_subjects)))
    subjects_split = {'train': training_subjects, 
                      'valid': valid_subjects, 
                      'test': test_subjects}

    user_level_texts = {}
    for row in writings_df.sort_values(by='date').itertuples():
        words = []
        raw_text = ""
        if hasattr(row, 'tokenized_title'):
            if row.tokenized_title:
                words.extend(row.tokenized_title)
                raw_text += row.title
        if hasattr(row, 'tokenized_text'):
            if row.tokenized_text:
                words.extend(row.tokenized_text)
                raw_text += row.text
        if not words or len(words)<min_post_len:
#             logger.debug(row.subject)
            continue
        if labelcol == 'label':
            label = row.label
        liwc_categs = [getattr(row, categ) for categ in categories]
        if row.subject not in user_level_texts.keys():
            user_level_texts[row.subject] = {}
            user_level_texts[row.subject]['texts'] = [words]
            user_level_texts[row.subject]['label'] = label
            user_level_texts[row.subject]['liwc'] = [liwc_categs]
            user_level_texts[row.subject]['raw'] = [raw_text]
        else:
            user_level_texts[row.subject]['texts'].append(words)
            user_level_texts[row.subject]['liwc'].append(liwc_categs)
            user_level_texts[row.subject]['raw'].append(raw_text)
            
    return user_level_texts, subjects_split, vocabulary


def load_embeddings(path, embedding_dim, voc):
    # random matrix with mean value = 0
    embedding_matrix = np.random.random((len(voc)+2, embedding_dim)) - 0.5 # voc + unk + pad value(0)
    cnt_inv = 0
    f = open(path, encoding='utf8')
    for i, line in enumerate(f):
#         print(i)
        values = line.split()
        word = ''.join(values[:-hyperparams_features['embedding_dim']])
        coefs = np.asarray(values[-hyperparams_features['embedding_dim']:], dtype='float32')
        word_i = voc.get(word)
        if word_i is not None:
            embedding_matrix[word_i] = coefs
            cnt_inv += 1
    f.close()

    print('Total %s word vectors.' % len(embedding_matrix))
    print('Words not found in embedding space %d' % (len(embedding_matrix)-cnt_inv))
 
    return embedding_matrix