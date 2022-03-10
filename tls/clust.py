"""
MIT License

Copyright (c) 2020 Demian Gholipour Ghalandari

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

import collections
import datetime
from typing import List

import markov_clustering as mc
import numpy as np
from scipy import sparse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from . import data, utils
from .knee import calculate_avg_sum, detect_knee_point


class ClusteringTimelineGenerator():
    def __init__(self,
                 language="english",
                 mode="eval",
                 vectorizer=None,
                 clusterer=None,
                 cluster_ranker=None,
                 summarizer=None,
                 clip_sents=5,
                 key_to_model=None,
                 unique_dates=True,
                 k=False,
                 l=False):

        self.clusterer = clusterer or TemporalMarkovClusterer()
        self.cluster_ranker = cluster_ranker or ClusterDateMentionCountRanker()
        self.summarizer = summarizer or summarizer.CentroidOpt()
        self.key_to_model = key_to_model
        self.unique_dates = unique_dates
        self.clip_sents = clip_sents

        self.language = language
        self.mode = mode
        self.k = k
        self.l = l
        self.vectorizer = vectorizer

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):

        print('clustering articles...')

        if not self.vectorizer:
            if self.language == "english":
                doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            else:
                stopwords = data.load_stopwords('./extra_data/cn_stopwords.txt')
                doc_vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords)

            articles = list(collection.articles())
            texts = ['{} {}'.format(a.tokenized_title, a.tokenized_text) for a in articles]
            doc_vectorizer.fit(texts)
        else:
            doc_vectorizer = self.vectorizer

        # print(len(list(collection.articles())))

        clusters = self.clusterer.cluster(collection, doc_vectorizer)

        print('assigning cluster times...')
        for c in clusters:
            c.time = c.most_mentioned_time()
            if c.time is None:
                c.time = c.earliest_pub_time()

        print('ranking clusters...')
        ranked_clusters, ranked_score = self.cluster_ranker.rank(clusters, collection)

        print('vectorizing sentences...')
        raw_sents = [s.token for a in collection.articles() for s in
                     a.sentences]  # [:self.clip_sents]]

        if not self.vectorizer:
            if self.language == "english":
                self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            else:
                stopwords = data.load_stopwords('./extra_data/cn_stopwords.txt')
                self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords)
            self.vectorizer.fit([s.token for a in collection.articles() for s in a.sentences])


        def sent_filter(sent):
            """
            Returns True if sentence is allowed to be in a summary.
            """
            lower = sent.token.lower()
            if not output_titles and sent.is_title:
                return False
            # elif not output_body_sents and not sent.is_sent:
            #     return False
            else:
                return True

        print('summarization...')
        sys_l = 0
        sys_m = 0
        if self.l:
            ranked_score = calculate_avg_sum(ranked_score)
            max_dates = detect_knee_point(ranked_score) + 1
        ref_m = max_dates * max_summary_sents
        date_to_summary = collections.defaultdict(list)
        for i, c in enumerate(ranked_clusters):

            date = c.time.date()
            c_sents = self._select_sents_from_cluster(c)
            # print("C", date, len(c_sents), "M", sys_m, "L", sys_l)
            if len(c_sents) <= 0:
                continue
            summary, summary_token = self.summarizer.summarize(
                c_sents,
                k=max_summary_sents,
                vectorizer=self.vectorizer,
                filters=sent_filter,
                knee=self.k
            )

            summary = summary_token if self.mode == "eval" else summary

            if summary:
                if self.unique_dates and date in date_to_summary:
                    continue
                date_to_summary[date] += summary
                sys_m += len(summary)
                if self.unique_dates:
                    sys_l += 1

            # if sys_m >= ref_m or sys_l >= max_dates:
            #     break
            if sys_l >= max_dates:
                break

        timeline = []
        for d, summary in date_to_summary.items():
            t = datetime.datetime(d.year, d.month, d.day)
            timeline.append((t, summary))
        timeline.sort(key=lambda x: x[0])

        return data.Timeline(timeline)

    def _select_sents_from_cluster(self, cluster):
        sents = []
        for a in cluster.articles:
            for s in a.sentences:  # [:self.clip_sents]:
                sents.append(s)
        return sents

    def load(self, ignored_topics):
        pass

    def train(self, collection, topic):
        pass

################################# CLUSTERING ###################################


class Cluster:
    def __init__(self, articles, vectors, centroid, time=None, id=None):
        self.articles = sorted(articles, key=lambda x: x.time)
        self.centroid = centroid
        self.id = id
        self.vectors = vectors
        self.time = time

    def __len__(self):
        return len(self.articles)

    def pub_times(self):
        return [a.time for a in self.articles]

    def earliest_pub_time(self):
        return min(self.pub_times())

    def most_mentioned_time(self):
        mentioned_times = []
        for a in self.articles:
            for s in a.sentences:
                for t in s.time:
                    mentioned_times.append(t)
        if mentioned_times:
            return collections.Counter(mentioned_times).most_common()[0][0]
        else:
            return None

    def update_centroid(self):
        X = sparse.vstack([sparse.coo_matrix(item) for item in self.vectors])
        self.centroid = sparse.csr_matrix.mean(X, axis=0)


class Clusterer():
    def cluster(self, collection, vectorizer) -> List[Cluster]:
        raise NotImplementedError


class OnlineClusterer(Clusterer):
    def __init__(self, max_days=1, min_sim=0.5):
        self.max_days = max_days
        self.min_sim = min_sim

    def cluster(self, collection, vectorizer) -> List[Cluster]:
        # build article vectors
        texts = ['{} {}'.format(a.tokenized_title, a.tokenized_text) for a in collection.articles]
        X = vectorizer.get_matrix(texts)

        id_to_vector = {}
        for a, x in zip(collection.articles(), X):
            id_to_vector[a.id] = x

        online_clusters = []

        for t, articles in collection.time_batches():
            for a in articles:

                # calculate similarity between article and all clusters
                x = id_to_vector[a.id]
                cluster_sims = []
                for c in online_clusters:
                    if utils.days_between(c.time, t) <= self.max_days:
                        centroid = c.centroid
                        sim = cosine_similarity(centroid, x)[0, 0]
                        cluster_sims.append(sim)
                    else:
                        cluster_sims.append(0)

                # assign article to most similar cluster (if over threshold)
                cluster_found = False
                if len(online_clusters) > 0:
                    i = np.argmax(cluster_sims)
                    if cluster_sims[i] >= self.min_sim:
                        c = online_clusters[i]
                        c.vectors.append(x)
                        c.articles.append(a)
                        c.update_centroid()
                        c.time = t
                        online_clusters[i] = c
                        cluster_found = True

                # initialize new cluster if no cluster was similar enough
                if not cluster_found:
                    new_cluster = Cluster([a], [x], x, t)
                    online_clusters.append(new_cluster)

        clusters = []
        for c in online_clusters:
            cluster = Cluster(c.articles, c.vectors)
            clusters.append(cluster)

        return clusters


class TemporalMarkovClusterer(Clusterer):
    def __init__(self, max_days=1):
        self.max_days = max_days

    def cluster(self, collection, vectorizer) -> List[Cluster]:
        articles = list(collection.articles())
        texts = ['{} {}'.format(a.tokenized_title, a.tokenized_text) for a in articles]
        X = vectorizer.get_matrix(texts)
        times = [a.time for a in articles]
        print(len(X))
        # print(X[0])
        print(X.shape)
        print(X.ndim)
        print(np.array([X[i] for i in range(2)]).shape)

        print('temporal graph...')
        S = self.temporal_graph(X, times)
        # print('S shape:', S.shape)
        print('run markov clustering...')
        result = mc.run_mcl(S)
        print('done')

        idx_clusters = mc.get_clusters(result)
        idx_clusters.sort(key=lambda c: len(c), reverse=True)

        print(f'times: {len(set(times))} articles: {len(articles)} '
              f'clusters: {len(idx_clusters)}')

        clusters = []
        for c in idx_clusters:
            c_vectors = [X[i] for i in c]
            c_articles = [articles[i] for i in c]
            Xc = sparse.vstack([sparse.coo_matrix(item) for item in c_vectors])
            centroid = sparse.csr_matrix(Xc.mean(axis=0))
            cluster = Cluster(c_articles, c_vectors, centroid=centroid)
            clusters.append(cluster)

        return clusters

    def temporal_graph(self, X, times):
        # times = [utils.strip_to_date(t) for t in times]
        time_to_ixs = collections.defaultdict(list)
        for i in range(len(times)):
            time_to_ixs[times[i]].append(i)

        n_items = X.shape[0]
        S = sparse.lil_matrix((n_items, n_items))
        start, end = min(times), max(times)
        total_days = (end - start).days + 1

        for n in range(total_days + 1):
            t = start + datetime.timedelta(days=n)
            window_size = min(self.max_days + 1, total_days + 1 - n)
            window = [t + datetime.timedelta(days=k) for k in range(window_size)]

            if n == 0 or len(window) == 1:
                indices = [i for t in window for i in time_to_ixs[t]]
                if len(indices) == 0:
                    continue

                X_n = sparse.vstack([sparse.coo_matrix(X[i]) for i in indices])
                S_n = cosine_similarity(X_n)
                n_items = len(indices)
                for i_x, i_n in zip(indices, range(n_items)):
                    for j_x, j_n in zip(indices, range(i_n + 1, n_items)):
                        S[i_x, j_x] = S_n[i_n, j_n]
            else:
                # prev is actually prev + new
                prev_indices = [i for t in window for i in time_to_ixs[t]]
                new_indices = time_to_ixs[window[-1]]

                if len(new_indices) == 0:
                    continue

                X_prev = sparse.vstack([sparse.coo_matrix(X[i]) for i in prev_indices])
                X_new = sparse.vstack([sparse.coo_matrix(X[i]) for i in new_indices])
                S_n = cosine_similarity(X_prev, X_new)
                n_prev, n_new = len(prev_indices), len(new_indices)
                for i_x, i_n in zip(prev_indices, range(n_prev)):
                    for j_x, j_n in zip(new_indices, range(n_new)):
                        S[i_x, j_x] = S_n[i_n, j_n]

        return sparse.csr_matrix(S)


############################### CLUSTER RANKING ################################


class ClusterRanker:
    def rank(self, clusters, collection, vectorizer):
        raise NotImplementedError


class ClusterSizeRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None):
        return sorted(clusters, key=len, reverse=True)


class ClusterDateMentionCountRanker(ClusterRanker):
    def rank(self, clusters, collection=None, vectorizer=None):
        date_to_count = collections.defaultdict(int)
        for a in collection.articles():
            for s in a.sentences:
                for d in s.time:
                    date_to_count[d.date()] += 1

        clusters = sorted(clusters, reverse=True, key=len)

        def get_count(c):
            t = c.most_mentioned_time()
            if t:
                return date_to_count[t.date()]
            else:
                return 0

        # clusters = sorted(clusters, reverse=True, key=get_count)
        #
        # return sorted(clusters, key=len, reverse=True)
        clusters = sorted(clusters, reverse=True, key=get_count)
        clusters = sorted(clusters, key=len, reverse=True)
        return clusters, list(map(get_count, clusters))

