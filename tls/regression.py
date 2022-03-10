import datetime
import math
from collections import Counter

import numpy
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from tilse.evaluation import util
from tilse.evaluation.rouge import RougeReimplementation

from tls import data, post_processing


class Regression():
    """
    Predicts timelines using regression of per-day ROUGE-1 F1 score.

    Represents sentences with length, number NEs, avg/sum tf-idf scores and
    lexical unigram features, and learns a regression model using
    sklearn's LinearRegression with default parameters except for
    normalize=True.

    When predicting, sentences are selected greedily respecting
    the constraints of maximum number of days in a timeline and maximum
    daily summary length.
    """

    def __init__(self,
                 language="english",
                 mode="eval",
                 key_to_model=None):
        self.language = language
        self.mode = mode
        self.model = LinearRegression(normalize=True)
        self.vectorizer = feature_extraction.DictVectorizer(sparse=False)
        self.key_to_model = key_to_model


    def train(self, dataset, timeline_to_evaluate=None):
        """
        Trains the model.
        For details on training, see the docstring of this class.
        Returns:
            Nothing, `self.model` is updated.
        """
        rouge = RougeReimplementation()
        self.model = LinearRegression(normalize=True)

        features = []
        f1_scores = []

        print("train ", timeline_to_evaluate, '.......')

        for t, collection in zip(dataset.topics, dataset.collections):
            if t == timeline_to_evaluate:
                continue
            reference_timelines = collection.timelines
            sum_tfidf, avg_tfidf = self.preprocess(collection, self.language)

            i = 0
            for a in collection.articles():
                for sent in a.sentences:
                    sent_processed = [sent.token_list]
                    if len(sent.time) > 0:
                        date = sent.get_date().date()
                    else:
                        date = sent.pub_time.date()

                    ref_temp = {}
                    for j, tl in enumerate(reference_timelines):
                        if date in tl.date_to_summaries:
                            ref_temp[j] = tl.date_to_summaries[date]
                        else:
                            continue
                    if len(ref_temp.keys()) == 0:
                        continue

                    ref_processed = {}

                    for j, sents in ref_temp.items():
                        ref_processed[j] = [[x for x in s.split()] for s in sents]

                    rouge_computed = rouge.score_summary(sent_processed, ref_processed)

                    if rouge_computed["rouge_1_p_count"] == 0:
                        prec = 0
                    else:
                        prec = rouge_computed["rouge_1_h_count"] / rouge_computed["rouge_1_p_count"]

                    if rouge_computed["rouge_1_m_count"] == 0:
                        rec = 0
                    else:
                        rec = rouge_computed["rouge_1_h_count"] / rouge_computed["rouge_1_m_count"]

                    f1 = util.get_f_score(prec, rec)

                    features.append(Regression._compute_features_for_sent(
                        sent, i, sum_tfidf, avg_tfidf))

                    f1_scores.append(float(f1))

                    i += 1
        vectorized = self.vectorizer.fit_transform(features)
        vectorized = numpy.array(vectorized)
        vectorized = numpy.nan_to_num(vectorized)
        self.model.fit(vectorized, f1_scores)

    def predict(self,
                collection,
                max_dates=3,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):
        """
        Predicts a timeline. For details on how the prediction works,
        see the docstring for this class.
        """

        all_sents = [s for a in collection.articles() for s in a.sentences]
        ranked_sentences = self._get_ranked_sentences(all_sents, collection)

        post_processed = post_processing.post_process(
            [all_sents[i] for i in reversed(ranked_sentences.argsort())],
            None,
            max_summary_sents,
            max_dates,
            start=collection.start,
            end=collection.end)

        date_to_summary = {}

        for date in post_processed:
            summary = [s.text for s in post_processed[date]]
            summary_token = [s.token for s in post_processed[date]]
            summary = summary_token if self.mode == "eval" else summary
            date_to_summary[date] = summary

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)

    def _get_ranked_sentences(self, all_sents, collection):
        features = []
        sum_tfidf, avg_tfidf = self.preprocess(collection, self.language)
        i = 0
        for sent in all_sents:
            features.append(Regression._compute_features_for_sent(sent, i, sum_tfidf, avg_tfidf))
        features = self.vectorizer.transform(features)
        ranked_sentences = self.model.predict(features)
        return ranked_sentences + math.fabs(min(ranked_sentences))

    def preprocess(self, collection, language):
        """
        Computes tf-idf scores for sentences by summing/averaging tf-idf scores
        of words in the sentences.

        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.

        Returns:
            A tuple (of two numpy.array objects) containing scores for
            sentences obtained by  summing/averaging tf-idf scores for words
            in the sentences.
        """
        return Regression._compute_sum_and_avg_tfidf(collection, language)

    @staticmethod
    def _compute_features_for_sent(sent, i, sum_tfidf, avg_tfidf):
        all_features = Counter(sent.token_list)

        all_features.update({
            "len": len(sent.token_list),
            "number_nes": sent.name_entity_num,
            "sum_tfidf": sum_tfidf[i],
            "avg_tfidf": avg_tfidf[i]
        })

        return all_features

    @staticmethod
    def _compute_sum_and_avg_tfidf(collection, language):

        raw_sents = [s.token for a in collection.articles() for s in
                     a.sentences]  # [:self.clip_sents]]

        if language == "english":
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        else:
            stopwords = data.load_stopwords('./extra_data/cn_stopwords.txt')
            vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords)
        vectorizer.fit(raw_sents)
        sent_term_matrix = vectorizer.transform(raw_sents)

        tf_idf_sum = sent_term_matrix.sum(axis=1).A1
        nonzero_entries = numpy.diff(sent_term_matrix.indptr)

        return tf_idf_sum, tf_idf_sum / nonzero_entries

    def load(self, ignored_topics):
        key = ' '.join(sorted(ignored_topics))
        if self.key_to_model:
            self.model.model = self.key_to_model[key]
