import copy
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class IndexBuilder:
    def __init__(self, messages_df, word2idx, w2v_matrix, seed_df, min_msg_length, max_msg_length, rnn_model):
        self.messages_df = copy.deepcopy(messages_df)
        self.word2idx = word2idx
        self.w2v_matrix = w2v_matrix
        self.seed_df = copy.deepcopy(seed_df)
        self.min_msg_length = min_msg_length
        self.max_msg_length = max_msg_length
        self.rnn_model = copy.copy(rnn_model)
        self._final_sentiments_dict = None
        self._scores_by_day_na = None
        self._seed_dict = None

    def _get_k_most_similar_terms(self, term, k):
        term_idx = self.word2idx[term]
        term_emb = self.w2v_matrix[term_idx, :]
        all_sims = np.apply_along_axis(self.cos_sim, 1, np.delete(self.w2v_matrix, 0, 0), b=term_emb)
        all_terms_to_sims = dict(zip(self.word2idx.keys(), all_sims))
        del all_terms_to_sims[term]
        top_k_terms_sims = sorted(all_terms_to_sims.items(),
                                  key=lambda x: x[1], reverse=True)[:k]
        return dict(top_k_terms_sims)

    def _get_seed_dictionary(self):
        self.seed_df['sentiment'] = self.seed_df['sentiment'].apply(lambda x: 1 if x == 'positive' else -1)
        self._seed_dict = dict(zip(self.seed_df['keyword'].values,
                                   self.seed_df['sentiment'].values))

    def _get_augmented_dictionary(self):
        crypto_words = list(self.word2idx.keys())
        if self._seed_dict is None:
            self._get_seed_dictionary()
        seed_words = list(self._seed_dict.keys())
        common_terms_seed_crypto = list(set(seed_words).intersection(crypto_words))
        # --- get similarities between seed and new words as well as sentiment scores of new words
        crypto_terms_sentiment_dict = defaultdict(float)
        crypto_terms_similarity_dict = defaultdict(float)
        crypto_terms_already_mapped_to_common = set()
        for common_term in common_terms_seed_crypto:
            # --- get top two similar terms not yet in the common list
            common_term_top_k_sims_with_words = self._get_k_most_similar_terms(common_term, 2)
            for term_sim in common_term_top_k_sims_with_words.items():
                term = term_sim[0]
                sim = term_sim[1]
                if sim > 0 and term not in common_terms_seed_crypto:
                    if term in crypto_terms_already_mapped_to_common:
                        new_sim = sim
                        old_sim = crypto_terms_similarity_dict[term]
                        if new_sim > old_sim:  # if larger similarity found for this term
                            # --- assign new similarity value
                            crypto_terms_similarity_dict[term] = new_sim
                            # --- assign new sentiment score value
                            new_sentiment = new_sim * self._seed_dict[common_term]
                            crypto_terms_sentiment_dict[term] = new_sentiment
                    else:
                        # --- assign new similarity value
                        crypto_terms_similarity_dict[term] = sim
                        # --- assign new sentiment score value
                        new_sentiment = sim * self._seed_dict[common_term]
                        crypto_terms_sentiment_dict[term] = new_sentiment
        # --- get full dictionary
        self._final_sentiments_dict = crypto_terms_sentiment_dict.copy()
        self._final_sentiments_dict.update(self._seed_dict)

    def add_classfn_sentiment(self):
        if self._scores_by_day_na is None:
            messages_df_na = self.messages_df[self.messages_df['sentiment'] == 999]
            all_messages_na = messages_df_na['message'].values
            all_indices_na = messages_df_na.index.values
            all_sequences_na_with_indices = [([self.word2idx.get(t) for t in msg if t in self.word2idx.keys()], label)
                                             for msg, label in zip(all_messages_na, all_indices_na)]
            all_sequences_na_msgs = [x[0] for x in all_sequences_na_with_indices if len(x[0]) > self.min_msg_length]
            all_sequences_na_indices = [x[1] for x in all_sequences_na_with_indices if len(x[0]) > self.min_msg_length]
            messages_na_filtered = all_sequences_na_msgs
            indices_na_filtered = all_sequences_na_indices
            all_seqs_na_padded = pad_sequences(messages_na_filtered, maxlen=self.max_msg_length,
                                               padding="post", truncating="post")
            predictions_na = self.rnn_model.predict(all_seqs_na_padded)
            predictions_na_actual = np.argmax(predictions_na, axis=1)
            predictions_na_actual = predictions_na_actual.reshape(-1, 1)
            predictions_na_actual = np.apply_along_axis(lambda x: x - 1 if x == 0 else x,
                                                        1, predictions_na_actual)
            self.messages_df.loc[indices_na_filtered, ['sentiment']] = predictions_na_actual
            self.messages_df = self.messages_df.replace({'sentiment': {0: -1}})
            self.messages_df = self.messages_df[self.messages_df['sentiment'] != 999]
            self._scores_by_day_na = self.messages_df.loc[:, ['date', 'sentiment']] \
                .groupby(by=['date'])['sentiment']\
                .agg(num_pos=lambda x: sum([1 for y in x if y == 1]),
                     num_neg=lambda x: sum([1 for y in x if y == -1]),
                     sentiment=lambda x: np.nansum(x))
            # --- adjust sentiment to make it positive everywhere
            sentiment_adjustment = abs(self._scores_by_day_na['sentiment'].min()) + 1
            self._scores_by_day_na["sentiment_classfn"] = self._scores_by_day_na["sentiment"] + sentiment_adjustment
            self._scores_by_day_na.drop(columns=["sentiment"], inplace=True)
        else:
            print("Classification-based sentiment already added")

    def add_classfn_score_1(self):
        if self._scores_by_day_na is None:
            self.add_classfn_sentiment()
        self._scores_by_day_na['classfn_score_1'] = np.log(
            self._scores_by_day_na['sentiment_classfn'] /
            self._scores_by_day_na['sentiment_classfn'].shift(1))

    def add_classfn_score_2(self):
        if self._scores_by_day_na is None:
            self.add_classfn_sentiment()
        self._scores_by_day_na['classfn_score_2'] = \
            self._scores_by_day_na.apply(self.get_log_oi, axis=1)

    def add_w2v_sentiment(self):
        if self._scores_by_day_na is None:
            self.add_classfn_sentiment()
        if self._final_sentiments_dict is None:
            self._get_augmented_dictionary()
        self.messages_df["sentiment_w"] = self.messages_df['message'] \
            .apply(self.get_sim_based_score, sentiments_dict=self._final_sentiments_dict)
        scores_by_day_w2v = self.messages_df.dropna().loc[:, ['date', 'sentiment_w']] \
            .groupby(by=['date']) \
            .agg(lambda x: np.nansum(x))
        # --- adjust sentiment to make it positive everywhere
        sentiment_adjustment = abs(scores_by_day_w2v['sentiment_w'].min()) + 1
        scores_by_day_w2v["sentiment_w2v"] = scores_by_day_w2v["sentiment_w"] + sentiment_adjustment
        self._scores_by_day_na = self._scores_by_day_na.join(scores_by_day_w2v.loc[:, ["sentiment_w2v"]],
                                                             how="left")

    @staticmethod
    def get_log_oi(x):
        index = np.log((1 + x.num_pos) / (1 + x.num_neg))
        return index

    @staticmethod
    def get_sim_based_score(x, sentiments_dict):
        common_terms = set(x).intersection(sentiments_dict.keys())
        if len(common_terms) > 0:
            sentiment = np.mean([sentiments_dict[w] for w in common_terms])
        else:
            sentiment = np.nan
        return sentiment

    @staticmethod
    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    @property
    def final_sentiments_dict(self):
        return self._final_sentiments_dict

    @property
    def scores_by_day_na(self):
        return self._scores_by_day_na

    @property
    def seed_dict(self):
        return self._seed_dict


