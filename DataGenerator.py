class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, user_level_data, subjects_split, session=None, use_bert=False, set_type='train',
                 batch_size=32, seq_len=512, vocabulary=vocabulary,
                 voc_size=hyperparams_features['max_features'], emotion_lexicon=nrc_lexicon,
                 hierarchical=False, pad_value=0, padding='pre',
                 post_groups_per_user=None, posts_per_group=10, post_offset = 0,
                 sampling_distr_alfa=0.1, sampling_distr='exp', # 'exp', 'uniform'
                 emotions=emotions, pronouns=["i", "me", "my", "mine", "myself"], liwc_categories=liwc_categories,
                 liwc_dict=liwc_dict, compute_liwc=False, liwc_words_for_categories=None,
                 pad_with_duplication=False,
                 max_posts_per_user=None, sample_seqs=True,
                 shuffle=True, return_subjects=False, keep_last_batch=True, class_weights=None,
                classes=1):
        'Initialization'
        self.seq_len = seq_len
        # Instantiate tokenizer
        if session:
            self.bert_tokenizer = create_tokenizer_from_hub_module(session)
            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
        else:
            if use_bert:
                logger.error("Need a session to use bert in data generation")
            self.bert_tokenizer = None
        self.use_bert = use_bert
        self.subjects_split = subjects_split
        self.set = set_type
        self.emotion_lexicon = emotion_lexicon
        self.batch_size = batch_size
        self.hierarchical = hierarchical
        self.data = user_level_data
        self.pad_value = pad_value
        self.return_subjects = return_subjects
        self.sampling_distr_alfa = sampling_distr_alfa
        self.sampling_distr = sampling_distr
        self.emotions = emotions
        self.pronouns = pronouns
        self.liwc_categories = liwc_categories
        self.liwc_dict = liwc_dict
        self.liwc_words_for_categories = liwc_words_for_categories
        self.compute_liwc = compute_liwc
        self.sample_seqs = sample_seqs
        self.pad_with_duplication = pad_with_duplication
        self.padding = padding
        self.keep_last_batch = keep_last_batch
        self.shuffle = shuffle
        self.voc_size = voc_size
        self.vocabulary = vocabulary
        self.max_posts_per_user = max_posts_per_user
        self.post_groups_per_user = post_groups_per_user
        self.post_offset = post_offset
        self.posts_per_group = posts_per_group
        self.classes = classes
        self.class_weights = class_weights
        self.generated_labels = []
        self.__post_indexes_per_user()
        self.on_epoch_end()
        
    @staticmethod
    def _random_sample(population_size, sample_size, sampling_distr, alfa=0.1, replacement=False):
        if sampling_distr == 'exp':
            # Exponential sampling
            sample = sorted(np.random.choice(population_size, 
                            min(sample_size, population_size),
                            p = DataGenerator.__generate_reverse_exponential_indices(population_size, alfa),
                            replace=replacement))
                                                                # if pad_with_duplication, 
                                                                # pad by adding the same post multiple times
                                                                # if there are not enough posts
        elif sampling_distr == 'uniform':
            # Uniform sampling
            sample = sorted(np.random.choice(population_size,
                            min(sample_size, population_size),
                            replace=replacement))
        return sample
    
    @staticmethod
    def __generate_reverse_exponential_indices(max_index, alfa=1):
        probabilities = []
        for x in range(max_index):
            probabilities.append(alfa * (np.exp(alfa*x)))
        reverse_probabilities = [p for p in probabilities]
        sump = sum(reverse_probabilities)
        normalized_probabilities = [p/sump for p in reverse_probabilities]
        return normalized_probabilities
    
    def __post_indexes_per_user(self):
        self.indexes_per_user = {u: [] for u in range(len(self.subjects_split[self.set]))}
        self.indexes_with_user = []
        self.item_weights = []
        for u in range(len(self.subjects_split[self.set])):
            if self.subjects_split[self.set][u] not in self.data:
                logger.warning("User %s has no posts in %s set. Ignoring.\n" % (
                    self.subjects_split[self.set][u], self.set))
                continue
            user_posts = self.data[self.subjects_split[self.set][u]]['texts']
            if self.max_posts_per_user:
                user_posts = user_posts[:self.max_posts_per_user]
            nr_post_groups = int(np.ceil(len(user_posts) / self.posts_per_group))
            
            if self.post_groups_per_user:
                nr_post_groups = min(self.post_groups_per_user, nr_post_groups)
            for i in range(nr_post_groups):
                # Generate random ordered samples of the posts
                if self.sample_seqs:
                    indexes_sample = DataGenerator._random_sample(population_size=len(user_posts),
                                                         sample_size=self.posts_per_group,
                                                         sampling_distr=self.sampling_distr,
                                                         alfa=self.sampling_distr_alfa,
                                                         replacement=self.pad_with_duplication)
                    self.indexes_per_user[u].append(indexes_sample)
                    self.indexes_with_user.append((u, indexes_sample))
                    # break # just generate one?
                # Generate all subsets of the posts in order
                # TODO: Change here if you want a sliding window
                else:
                    self.indexes_per_user[u].append(range(i*self.posts_per_group + self.post_offset,
                                                        min((i+1)*self.posts_per_group + self.post_offset, len(user_posts))))
                    self.indexes_with_user.append((u, range(i*self.posts_per_group ,
                                                        min((i+1)*self.posts_per_group + self.post_offset, len(user_posts)))))

        if self.class_weights:
            for item in self.indexes_with_user:
                u, _ = item
                s = self.subjects_split[self.set][u]
                # Note: weight 1 is default.
                try:
                    self.item_weights.append(1./self.class_weights[self.data[s]['label']])
                except Exception as e:
                    self.item_weights.append(1)
                    logger.error("Could not compute item weight for user %s. " % s + str(e) + "\n")
        else:
            self.item_weights = []

    def __encode_text(self, tokens, raw_text):
        # Using voc_size-1 value for OOV token
        encoded_tokens = [self.vocabulary.get(w, self.voc_size-1) for w in tokens]
        encoded_emotions = encode_emotions(tokens, self.emotion_lexicon, self.emotions)
        encoded_pronouns = encode_pronouns(tokens, self.pronouns)
        encoded_stopwords = encode_stopwords(tokens)
        if not self.compute_liwc:
            encoded_liwc = None
        else:
            encoded_liwc = self.__encode_liwc_categories(tokens)
        if self.bert_tokenizer:
            bert_ids, bert_masks, bert_segments, label = encode_text_for_bert(self.bert_tokenizer, InputExample(None, 
                                               raw_text), self.seq_len)
        else:
            bert_ids, bert_masks, bert_segments = [[0]*self.seq_len, [0]*self.seq_len, [0]*self.seq_len]
        return (encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc,
               bert_ids, bert_masks, bert_segments)
    
    def __encode_liwc_categories_full(self, tokens, relative=True):
        categories_cnt = [0 for c in self.liwc_categories]
        if not tokens:
            return categories_cnt
        text_len = len(tokens)
        for i, category in enumerate(self.liwc_categories):
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
        
        
    def __encode_liwc_categories(self, tokens, relative=True):
        categories_cnt = [0 for c in self.liwc_categories]
        if not tokens:
            return categories_cnt
        text_len = len(tokens)
        for i, category in enumerate(self.liwc_categories):
            for t in tokens:
                if t in self.liwc_words_for_categories[category]:
                    categories_cnt[i] += 1
            if relative and text_len:
                categories_cnt[i] = categories_cnt[i]/text_len
        return categories_cnt
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.keep_last_batch:
            return int(np.ceil(len(self.indexes) / self.batch_size)) # + 1 to not discard last batch
        return int((len(self.indexes))/self.batch_size)
        
    def __getitem__(self, index):
        'Generate one batch of data'
        # Reset generated labels
        if index == 0:
             self.generated_labels = []
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find users
        user_indexes = [t[0] for t in indexes]
        users = set([self.subjects_split[self.set][i] for i in user_indexes
                    if self.subjects_split[self.set][i] in self.data.keys()]) # TODO: maybe needs a warning that user is missing
        post_indexes_per_user = {u: [] for u in users}
        # Sample post ids
        for u, post_indexes in indexes:
            user = self.subjects_split[self.set][u]
            # Note: was bug here - changed it into a list
            post_indexes_per_user[user].append(post_indexes)

        # Generate data
        if self.hierarchical:
            X, s, y = self.__data_generation_hierarchical(users, post_indexes_per_user)
        else:
            X, s, y = self.__data_generation(users, post_indexes_per_user)

        if self.return_subjects:
            return X, s, y
        else:
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.indexes_with_user
#         np.arange(len(self.subjects_split[self.set]))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        if self.class_weights:
            # Sample users according to class weight (Or do this for each batch instead?)
            normalized_weights = [w/sum(self.item_weights) for w in self.item_weights]
            random_user_indexes = np.random.choice(len(self.indexes_with_user), 
                            len(self.indexes_with_user),
                            p = normalized_weights, replace=True)
            self.indexes = [self.indexes_with_user[i] for i in random_user_indexes]
    def __data_generation(self, users, post_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        tokens_data = []
        categ_data = []
        sparse_data = []
        subjects = []
        bert_ids_data = []
        bert_masks_data = []
        bert_segments_data = []
        labels = []

        for subject in users:

            if 'label' in self.data[subject]:
                label = self.data[subject]['label']
            else:
                label = None

            
            all_words = []
            all_raw_texts = []
            liwc_aggreg = []

            for post_index_range in post_indexes[subject]:
                # Sample
                texts = [self.data[subject]['texts'][i] for i in post_index_range]
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_selection = [self.data[subject]['liwc'][i] for i in post_index_range]
                raw_texts = [self.data[subject]['raw'][i] for i in post_index_range]

                all_words.append(sum(texts, [])) # merge all texts in group in one list
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_aggreg.append(np.array(liwc_selection).mean(axis=0).tolist())
                all_raw_texts.append(" ".join(raw_texts))
            for i, words in enumerate(all_words):
                encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, \
                    bert_ids, bert_masks, bert_segments = self.__encode_text(words, all_raw_texts[i])
                try:
                    subject_id = int(re.findall('[0-9]+', subject)[0])
                except IndexError:
                    subject_id = subject
                tokens_data.append(encoded_tokens)
                # TODO: what will be the difference between these?
                # I think instead of averaging for the post group, it just does it correctly
                # for the whole post group (when computing, non-lazily)
                if 'liwc' in self.data[subject] and not self.compute_liwc:  
                    categ_data.append(encoded_emotions + [encoded_pronouns] + liwc_aggreg[i])
                   
                else:
                    categ_data.append(encoded_emotions + [encoded_pronouns] + encoded_liwc)
                    
                sparse_data.append(encoded_stopwords)
                bert_ids_data.append(bert_ids)
                bert_masks_data.append(bert_masks)
                bert_segments_data.append(bert_segments)
                
                labels.append(label)
                subjects.append(subject_id)

        
        self.generated_labels.extend(labels)
        # using zeros for padding
        tokens_data_padded = sequence.pad_sequences(tokens_data, maxlen=self.seq_len, 
                                                    padding=self.padding,
                                                   truncating=self.padding)

        if self.use_bert:
            return ([np.array(tokens_data_padded), np.array(categ_data), np.array(sparse_data),
                 np.array(bert_ids_data), np.array(bert_masks_data), np.array(bert_segments_data),
                ],
                np.array(subjects),
                np.array(labels))
        else:
            return ([np.array(tokens_data_padded), np.array(categ_data), np.array(sparse_data),
                ],
                np.array(subjects),
                np.array(labels))
    
    def __data_generation_hierarchical(self, users, post_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        user_tokens = []
        user_categ_data = []
        user_sparse_data = []
        user_bert_ids_data = []
        user_bert_masks_data = []
        user_bert_segments_data = []
        
        labels = []
        subjects = []
        for subject in users:
             
            all_words = []
            all_raw_texts = []
            liwc_scores = []
            
            if 'label' in self.data[subject]:
                if self.classes==1:
                    label = self.data[subject]['label']
                else:
                    label = list(np_utils.to_categorical(self.data[subject]['label'], num_classes=self.classes))
            else:
                label = None

            for post_index_range in post_indexes[subject]:
                # Sample
                texts = [self.data[subject]['texts'][i] for i in post_index_range]
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_selection = [self.data[subject]['liwc'][i] for i in post_index_range]
                raw_texts = [self.data[subject]['raw'][i] for i in post_index_range]

                all_words.append(texts)
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_scores.append(liwc_selection)
                all_raw_texts.append(raw_texts)
            
#             if len(texts) < self.max_posts_per_user:
#                 # TODO: pad with zeros
#                 pass

            for i, words in enumerate(all_words):
                tokens_data = []
                categ_data = []
                sparse_data = []
                bert_ids_data = []
                bert_masks_data = []
                bert_segments_data = []
                
                raw_text = all_raw_texts[i]
                words = all_words[i]
                
                for p, posting in enumerate(words): 
                    encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, \
                        bert_ids, bert_masks, bert_segments = self.__encode_text(words[p], raw_text[p])
                    if 'liwc' in self.data[subject] and not self.compute_liwc:
                        liwc = liwc_scores[i][p]
                    else:
                        liwc = encoded_liwc
                    try:
                        subject_id = int(re.findall('[0-9]+', subject)[0])
                    except IndexError:
                        subject_id = subject
                    tokens_data.append(encoded_tokens)
                    # using zeros for padding
                    # TODO: there is something wrong with this
                    categ_data.append(encoded_emotions + [encoded_pronouns] + liwc)
                    sparse_data.append(encoded_stopwords)
                    bert_ids_data.append(bert_ids)
                    bert_masks_data.append(bert_masks)
                    bert_segments_data.append(bert_segments)
                
                # For each range
                tokens_data_padded = np.array(sequence.pad_sequences(tokens_data, maxlen=self.seq_len,
                                              padding=self.padding,
                                            truncating=self.padding))
                user_tokens.append(tokens_data_padded)

                user_categ_data.append(categ_data)
                user_sparse_data.append(sparse_data)

                user_bert_ids_data.append(bert_ids_data)
                user_bert_masks_data.append(bert_masks_data)
                user_bert_segments_data.append(bert_segments_data)

                labels.append(label)
                subjects.append(subject_id)

        user_tokens = sequence.pad_sequences(user_tokens, 
                                             maxlen=self.posts_per_group, 
                                             value=self.pad_value)
        user_tokens = np.rollaxis(np.dstack(user_tokens), -1)
        user_categ_data = sequence.pad_sequences(user_categ_data,  
                                                 maxlen=self.posts_per_group, 
                                                 value=self.pad_value, dtype='float32')
        user_categ_data = np.rollaxis(np.dstack(user_categ_data), -1)
        
        user_sparse_data = sequence.pad_sequences(user_sparse_data, 
                                                  maxlen=self.posts_per_group, 
                                                  value=self.pad_value)
        user_sparse_data = np.rollaxis(np.dstack(user_sparse_data), -1)
        
        user_bert_ids_data = sequence.pad_sequences(user_bert_ids_data, 
                                                    maxlen=self.posts_per_group, 
                                                    value=self.pad_value)
        user_bert_ids_data = np.rollaxis(np.dstack(user_bert_ids_data), -1)
        
        user_bert_masks_data = sequence.pad_sequences(user_bert_masks_data, 
                                                      maxlen=self.posts_per_group, 
                                                      value=self.pad_value)
        user_bert_masks_data = np.rollaxis(np.dstack(user_bert_masks_data), -1)
        
        user_bert_segments_data = sequence.pad_sequences(user_bert_segments_data, 
                                                         maxlen=self.posts_per_group, 
                                                         value=self.pad_value)
        user_bert_segments_data = np.rollaxis(np.dstack(user_bert_segments_data), -1)

        self.generated_labels.extend(labels)

        labels = np.array(labels)
        
        if self.use_bert:
            return ((user_tokens, user_categ_data, user_sparse_data, 
                 user_bert_ids_data, user_bert_masks_data, uifser_bert_segments_data),
                np.array(subjects),
                labels)
        else:
            return ((user_tokens, user_categ_data, user_sparse_data), 
                np.array(subjects),
                labels)