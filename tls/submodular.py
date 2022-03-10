"""
The MIT License (MIT)

Copyright (c) 2017-2018 Sebastian Martschat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import datetime
import math
from collections import defaultdict, Counter

import numpy
from numba import jit
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import data


class Submodular():
    """
    Predicts timelines using submodular optimization.

    Timelines are constructed using a greedy algorithm optimizing a submodular
    objective function under suitable constraints.

    For more details, see Martschat and Markert (CoNLL 2018): A temporally
    sensitive submodularity framework for timeline summarization.
    """

    def __init__(self,
                 language="english",
                 mode="eval",
                 vectorizer=None):
        self.language = language
        self.mode = mode
        self.vectorizer = vectorizer

    def predict(self,
                collection,
                max_dates=3,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):

        (coeff_coverage,
         coeff_semantic_redundancy,
         coeff_date_redundancy,
         coeff_date_references) = 1, 1, 0, 1

        (coverage_values,
         sent_cluster_indices_semantic,
         sent_cluster_indices_date,
         sent_date_indices,
         date_references,
         singleton_rewards_semantic,
         singleton_rewards_date) = self.preprocess(collection)

        all_sent_dates = []

        all_sents = [s for a in collection.articles() for s in a.sentences]

        for sent in all_sents:
            if len(sent.time) > 0:
                date = sent.get_date().date()
            else:
                date = sent.pub_time.date()
            all_sent_dates.append(date)

            # greedy algorithm
        date_to_sent_mapping = defaultdict(list)
        selected_sent_indices = list()
        unselected_sent_indices = list(range(len(all_sents)))
        candidate_indices = [k for k in unselected_sent_indices
                             if is_valid_individual_constraints(k,
                                                                date_to_sent_mapping,
                                                                all_sent_dates,
                                                                collection,
                                                                max_dates,
                                                                max_summary_sents)]

        dates_selected = numpy.zeros(len(date_references))

        # contains partially precomputed per-cluster sums to facilitate greedy
        # algorithm diversity difference computation
        sums_semantic = numpy.zeros(max(sent_cluster_indices_semantic) + 1)
        sums_date = numpy.zeros(max(sent_cluster_indices_date) + 1)

        while candidate_indices:
            # numba workaround (cannot handle empty lists)
            if not selected_sent_indices:
                selected_sent_indices.append(-1)

            index, val = _objective_function(
                candidate_indices,
                coverage_values,
                singleton_rewards_semantic,
                singleton_rewards_date,
                sent_cluster_indices_semantic,
                sent_cluster_indices_date,
                sums_semantic,
                sums_date,
                sent_date_indices,
                dates_selected,
                date_references,
                coeff_coverage,
                coeff_semantic_redundancy,
                coeff_date_redundancy,
                coeff_date_references
            )

            if val >= 0:
                selected_sent_indices.append(index)
                date_to_sent_mapping[all_sent_dates[index]].append(all_sents[index])

                sums_semantic[
                    sent_cluster_indices_semantic[index]
                ] += singleton_rewards_semantic[index]

                sums_date[sent_cluster_indices_date[index]] += singleton_rewards_date[index]

                dates_selected[sent_date_indices[index]] = 1

            # numba workaround
            if selected_sent_indices[0] == -1:
                selected_sent_indices = selected_sent_indices[1:]

            unselected_sent_indices.remove(index)

            candidate_indices = [k for k in unselected_sent_indices if
                                 is_valid_individual_constraints(k,
                                                                 date_to_sent_mapping,
                                                                 all_sent_dates,
                                                                 collection,
                                                                 max_dates,
                                                                 max_summary_sents)]

        date_to_summary = {}

        for date in date_to_sent_mapping:
            summary = [s.text for s in date_to_sent_mapping[date]]
            summary_token = [s.token for s in date_to_sent_mapping[date]]
            summary = summary_token if self.mode == "eval" else summary
            date_to_summary[date] = summary

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)

    def preprocess(self, collection):
        """
        Computes various information for use in the objective function. For
        details, see below.

        Params:
            topic_name (str): name of the topic to which the corpus belongs.
            corpus (tilse.data.corpora.Corpus): A corpus.

        Returns:
            A 7-tuple containing:
                coverage_values (numpy.array): Coverage values for all
                    sentences.
                sent_cluster_indices_semantic (list(int)): Sentence cluster
                    indices for semantic cluster function.
                sent_cluster_indices_date (list(int)): Sentence cluster
                    indices for date cluster function.
                sent_date_indices list(int): Date index for each sentence (two
                    sentences have the same date iff they have the same index).
                date_references (numpy.array): For each date (represented by
                    its index), the number of references to it in the corpus.
                singleton_rewards_semantic (numpy.array): Singleton rewards
                    for each sentence according to semantic cluster function.
                singleton_rewards_date (numpy.array): Singleton rewards
                    for each sentence according to date cluster function.
        """
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

        sent_cluster_indices_semantic = clusters_by_similarity(X)
        sent_cluster_indices_date = clusters_by_date(sents)

        # assign indices to dates
        date_to_index = self._get_date_to_index_mapping(sents)

        # map sentence indicies to indices of their dates
        sent_date_indices = numpy.zeros(len(sents), dtype=int)
        for i, sent in enumerate(sents):
            if len(sent.time) > 0:
                date = sent.get_date().date()
            else:
                date = sent.pub_time.date()
            sent_date_indices[i] = date_to_index[date]

        date_references = numpy.zeros(len(date_to_index))

        dates_that_are_referred_to = []

        for sent in sents:
            if len(sent.time) > 0:
                date = sent.get_date().date()
            else:
                date = sent.pub_time.date()
            if date != sent.pub_time:
                # found reference to sent.date
                dates_that_are_referred_to.append(date)

        dates_with_frequency = Counter(dates_that_are_referred_to)

        # maps date index to how often date was referred to
        for date, i in date_to_index.items():
            date_references[i] = dates_with_frequency[date]

        singleton_rewards_semantic = numpy.zeros(len(sents))
        singleton_rewards_date = numpy.zeros(len(sents))

        # compute singleton reward of each sentence as total similarity to
        # all sentences
        for i in range(len(sents)):
            for j in range(len(sents)):
                singleton_rewards_semantic[i] += similarities[i, j]
                singleton_rewards_date[i] += similarities[i, j]

        sims = similarities / similarities.sum()
        date_references = date_references / date_references.sum()

        coverage_values = numpy.sum(sims, axis=1)

        cluster_to_sents_sum_semantic = defaultdict(float)
        cluster_to_sents_sum_date = defaultdict(float)
        for i, clust in enumerate(sent_cluster_indices_semantic):
            cluster_to_sents_sum_semantic[clust] += singleton_rewards_semantic[i]

        for i, clust in enumerate(sent_cluster_indices_date):
            cluster_to_sents_sum_date[clust] += singleton_rewards_date[i]

        # print(singleton_rewards_semantic[:10])
        # print(list(cluster_to_sents_sum_semantic)[:10])
        singleton_rewards_semantic = singleton_rewards_semantic / sum(
            [math.sqrt(float(x)) for x in list(cluster_to_sents_sum_semantic)]
        )
        singleton_rewards_date = singleton_rewards_date / sum(
            [math.sqrt(float(x)) for x in list(cluster_to_sents_sum_date)]
            # [math.sqrt(x) for x in cluster_to_sents_sum_date.values()]
        )

        return (coverage_values,
                sent_cluster_indices_semantic,
                sent_cluster_indices_date,
                sent_date_indices,
                date_references,
                singleton_rewards_semantic,
                singleton_rewards_date)

    def _get_date_to_index_mapping(self, sents):
        date_to_index = {}
        i = 0
        for sent in sents:
            if len(sent.time) > 0:
                date = sent.get_date().date()
            else:
                date = sent.pub_time.date()
            if date not in date_to_index:
                date_to_index[date] = i
                i += 1

        return date_to_index

    def load(self, ignored_topics):
        pass

    def train(self, collection, topic):
        pass


def clusters_by_similarity(all_sents_vectors):
    num_clusters = int(0.2 * all_sents_vectors.shape[0])
    if num_clusters == 0:
        num_clusters = 1
    kmeans = cluster.MiniBatchKMeans(num_clusters, random_state=23)
    kmeans.fit_predict(all_sents_vectors)

    return kmeans.labels_


def clusters_by_date(sentences):
    dates_to_index = {}
    labels = []
    for sent in sentences:
        if len(sent.time) > 0:
            date = sent.get_date().date()
        else:
            date = sent.pub_time.date()

        if date not in dates_to_index:
            if not dates_to_index:
                dates_to_index[date] = 0
            else:
                dates_to_index[date] = max(dates_to_index.values()) + 1

        labels.append(dates_to_index[date])
    return labels


def is_valid_total_length(index,
                          date_to_sent_mapping,
                          all_sent_dates,
                          collection):
    """
    Checks whether adding the sentence in focus would not violate
    the constraint of limiting the total number of sentences in a
    timeline.

    Corresponds to the AsMDS constrained described in Martschat and Markert
    (2018).

    Params:
        index (int): Index of the sentence in focus.
        date_to_sent_mapping (dict(datetime.datetime, list(tilse.data.sentences.Sentence)):
            Mapping of dates in a timeline to the sentences in the summary for this date
            (describes the partial timeline constructed so far).
        all_sent_dates (list(datetime.dateime)): Dates for all sentences. In particular,
            `all_sent_dates[index]` is the date of the sentence in focus.
        timeline_properties (tilse.models.timeline_properties.TimelineProperties):
            Properties of the timeline to predict.

    Returns:
        False if (i) date of the sentence in focus is before start or after end date of the
        timeline as defined in `timeline_properties` or (ii) adding the sentence in focus
        would lead to a timeline with more sentences than `timeline_properties.num_sentences`;
        True otherwise.
    """
    selected_date = all_sent_dates[index]

    if selected_date < collection.start.date() \
            or selected_date > collection.end.date():
        return False

    return sum([len(sents) for sents in date_to_sent_mapping.values()]) \
           < collection.num_sentences


@jit(nopython=True)
def _objective_function(candidate_indices,
                        coverage_values,
                        singleton_rewards_semantic,
                        singleton_rewards_date,
                        sent_cluster_indices_semantic,
                        sent_cluster_indices_date,
                        sums_semantic,
                        sums_date,
                        sent_date_indices,
                        dates_selected,
                        date_references,
                        coeff_coverage,
                        coeff_semantic_redundancy,
                        coeff_date_redundancy,
                        coeff_date_references):
    best = -1
    best_val = -numpy.inf

    for i in candidate_indices:
        # coverage
        my_sum = coeff_coverage * coverage_values[i]

        # redundancy...

        # ...via semantic clusters
        cluster_of_sent_semantic = sent_cluster_indices_semantic[i]
        sum_before = sums_semantic[cluster_of_sent_semantic]
        my_sum += coeff_semantic_redundancy * (
                math.sqrt(sum_before + singleton_rewards_semantic[i]) - math.sqrt(sum_before)
        )

        # ...via date clusters
        cluster_of_sent_date = sent_cluster_indices_date[i]
        sum_before = sums_date[cluster_of_sent_date]
        my_sum += coeff_date_redundancy * (
                math.sqrt(sum_before + singleton_rewards_date[i]) - math.sqrt(sum_before)
        )

        # date references
        date_index = sent_date_indices[i]
        if dates_selected[date_index] == 0:
            my_sum += coeff_date_references * date_references[date_index]

        # update best
        if my_sum > best_val:
            best = i
            best_val = my_sum

    return best, best_val


def is_valid_individual_constraints(index,
                                    date_to_sent_mapping,
                                    all_sent_dates,
                                    collection,
                                    max_dates=3,
                                    max_summary_sents=1,
                                    ):
    """
    Checks whether adding the sentence in focus would not violate
    the constraint of limiting the number of days and the length
    of daily summaries in the timeline.

    Corresponds to the TLSConstraints constraints described in Martschat and
    Markert (2018).

    Params:
        index (int): Index of the sentence in focus.
        date_to_sent_mapping (dict(datetime.datetime, list(tilse.data.sentences.Sentence)):
            Mapping of dates in a timeline to the sentences in the summary for this date
            (describes the partial timeline constructed so far).
        all_sent_dates (list(datetime.dateime)): Dates for all sentences. In particular,
            `all_sent_dates[index]` is the date of the sentence in focus.
        timeline_properties (tilse.models.timeline_properties.TimelineProperties):
            Properties of the timeline to predict.

    Returns:
        False if (i) date of the sentence in focus is before start or after end date of the
        timeline as defined in `timeline_properties` or (ii) adding the sentence in focus
        would lead to a timeline with more dates than `timeline_properties.num_dates` or
        (iii) adding the sentence in focus would lead to a timeline that has a daily summary
        longer than `timeline_properties.daily_summary_length` sentences; True otherwise.
    """

    summary_length = max_summary_sents
    desired_timeline_length = max_dates

    selected_date = all_sent_dates[index]

    if selected_date < collection.start.date() \
            or selected_date > collection.end.date():
        return False
    elif len(date_to_sent_mapping) == desired_timeline_length \
            and selected_date not in date_to_sent_mapping:
        return False
    elif selected_date in date_to_sent_mapping \
            and len(date_to_sent_mapping[selected_date]) == summary_length:
        return False
    else:
        return True
