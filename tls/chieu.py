
import datetime

import numpy
from numba import jit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import data, post_processing
from .knee import calculate_avg_sum, detect_knee_point


class Chieu():
    """
    Predicts timelines using the model of Chieu and Lee (2004): Query-based
    event extraction along a timeline.
    """

    def __init__(self,
                 language="english",
                 mode="eval",
                 vectorizer=None,
                 n=False):
        self.language = language
        self.mode = mode
        self.vectorizer = vectorizer

        self.n = n

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):

        raw_sents = [s.token for a in collection.articles() for s in a.sentences]
        sents = [s for a in collection.articles() for s in a.sentences]

        if not self.vectorizer:
            if self.language == "english":
                self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            else:
                stopwords = data.load_stopwords('./extra_data/cn_stopwords.txt')
                self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords)
            self.vectorizer.fit([s.token for a in collection.articles() for s in a.sentences])

        try:
            X = self.vectorizer.get_matrix(raw_sents)
        except:
            return None
        similarities = cosine_similarity(X)

        dates_in_ordinal = self._get_dates_to_ordinal(sents)
        date_diffs = self._get_date_diffs(dates_in_ordinal)
        sentence_ranks, extents = interest(similarities, date_diffs)

        if self.n:
            ranked_score = calculate_avg_sum(sentence_ranks)
            max_summary_sents = detect_knee_point(ranked_score) + 1

        post_processed = post_processing.post_process(
            [sents[i] for i in reversed(sentence_ranks.argsort())],
            [extents[i] for i in reversed(sentence_ranks.argsort())],
            max_summary_sents,  # daily summary length
            max_dates,  # timeline length
            start=collection.start,
            end=collection.end,
            knee=self.n
        )
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

    @staticmethod
    def _get_dates_to_ordinal(raw_sents):
        dates_in_ordinal = []
        for sent in raw_sents:
            if len(sent.time) > 0:
                dates_in_ordinal.append(sent.get_date().date())
            else:
                dates_in_ordinal.append(sent.pub_time.date())

        return numpy.array(dates_in_ordinal)

    @staticmethod
    @jit
    def _get_date_diffs(dates_in_ordinal):
        date_diffs = numpy.zeros((len(dates_in_ordinal), len(dates_in_ordinal)),
                                 dtype=numpy.uint32)

        for i in range(len(dates_in_ordinal)):
            for j in range(0, i):
                date1 = dates_in_ordinal[i]
                date2 = dates_in_ordinal[j]
                diff = abs(date1 - date2).days
                date_diffs[i][j] = diff
                date_diffs[j][i] = diff

        return date_diffs

    def load(self, ignored_topics):
        pass

    def train(self, collection, topic):
        pass

@jit(nopython=True)
def interest(similarities, date_diffs):
    """
        Computes interest as specified in Chieu and Lee (2004).

        The only differences is that we do not consider a reweighting of
        similarities by the "time span" of dates expressed in sentences (e.g.
        division by 30 if a sentence contains a reference to a month instead of a
        day), since we found the result of this operation to be insignificant,
        while it made the algorithm and computation more complex.

        Params:
            similarities (numpy.array): A matrix of sentence similarities.
            date_diffs (numpy.array): A matrix of date differences of sentences
            (e.g. difference between Sep 1 2001 and Sep 3 2001 is 2).

        Returns:
            A tuple consisting of a list of sentence ids with ranks (list(
            float)), and extents for the sentences (list(int)).
        """
    sentence_ranks = numpy.zeros(similarities.shape[0], dtype=numpy.uint16)
    extents = numpy.zeros(similarities.shape[0], dtype=numpy.uint16)

    interests = numpy.zeros((similarities.shape[0], 11), dtype=numpy.float32)

    for i in range(0, similarities.shape[0]):
        for j in range(0, similarities.shape[1]):
            for diff in range(1, 11):
                if date_diffs[i, j] <= diff:
                    interests[i][diff] += similarities[i, j]

        for diff in range(1, 11):
            if interests[i][diff] >= 0.8 * interests[i][10]:
                extents[i] = diff
                break

        sentence_ranks[i] = interests[i][10]

    return sentence_ranks, extents
