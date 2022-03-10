"""
@author: Li Xi
@file: data.py
@time: 2021/07/11 17:45
@desc:
"""

import collections
import datetime
import random

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from . import data, summarizers
from .knee import detect_knee_point, calculate_avg_sum

random.seed(42)


class OracleTimelineGenerator():
    def __init__(self,
                 language="english",
                 oracle_type="full", #"date", "text", "fulle"
                 mode="eval",
                 vectorizer=None,
                 date_ranker=None,
                 summarizer=None,
                 sent_collector=None,
                 clip_sents=5,
                 pub_end=2,
                 key_to_model=None,
                 k=False,
                 l=False):

        self.date_ranker = date_ranker
        self.sent_collector = sent_collector
        self.summarizer = summarizer
        self.key_to_model = key_to_model

        self.language = language
        self.mode = mode

        self.oracle_type = oracle_type
        self.vectorizer = vectorizer

        self.k = k
        self.l = l

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):
        print('vectorizer...')
        if not self.vectorizer:
            if self.language == "english":
                self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            else:
                stopwords = data.load_stopwords('./extra_data/cn_stopwords.txt')
                self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords)
            self.vectorizer.fit([s.token for a in collection.articles() for s in a.sentences])

        ref_dates = ref_tl.get_dates()
        # print(ref_dates)

        # ============ date selection && candidate sentences selection ============
        # ground truth date
        print('candidates & summarization...')
        if self.oracle_type == "date" or self.oracle_type == "full":
            dates_with_sents = self.sent_collector.collect_sents(
                ref_dates,
                collection,
                self.vectorizer,
                include_titles=input_titles,
            )
        else:
            # regression dates
            ranked_dates, ranked_score = self.date_ranker.rank_dates(collection)
            start = collection.start.date()
            end = collection.end.date()
            ranked = [(d, s) for d, s in zip(ranked_dates, ranked_score) if start <= d <= end]
            ranked_dates = [x[0] for x in ranked]
            ranked_score = [x[1] for x in ranked]
            print('candidates & summarization...')
            dates_with_sents = self.sent_collector.collect_sents(
                ranked_dates,
                collection,
                self.vectorizer,
                include_titles=input_titles,
            )

        # ============ summary generation ============

        def sent_filter(sent):
            """
            Returns True if sentence is allowed to be in a summary.
            """
            lower = sent.token.lower()
            # if not any([kw in lower for kw in collection.keywords]):
            #     return False
            if not output_titles and sent.is_title:
                return False
            elif not output_body_sents and not sent.is_sent:
                return False
            else:
                return True


        # date oracle: dates_with_sents_oracle + centroid opt
        if self.oracle_type == "text" or self.oracle_type == "full":
            # rouge
            timeline = []
            l = 0
            for i, (d, d_sents) in enumerate(dates_with_sents):
                if l >= max_dates:
                    break
                ref_summary = ref_tl.dates_to_summaries[d]
                # print(ref_summary)
                summary, summary_token = self.summarizer.summarize(
                    d_sents,
                    k=max_summary_sents,
                    vectorizer=self.vectorizer,
                    filters=sent_filter,
                    knee=self.k,
                    ref_tl=ref_summary
                )
                summary = summary_token if self.mode == "eval" else summary

                if summary:
                    time = datetime.datetime(d.year, d.month, d.day)
                    timeline.append((time, summary))
                    l += 1
        else:
            timeline = []
            l = 0
            for i, (d, d_sents) in enumerate(dates_with_sents):
                if l >= max_dates:
                    break

                summary, summary_token = self.summarizer.summarize(
                    d_sents,
                    k=max_summary_sents,
                    vectorizer=self.vectorizer,
                    filters=sent_filter,
                    knee=self.k
                )

                summary = summary_token if self.mode == "eval" else summary

                if summary:
                    time = datetime.datetime(d.year, d.month, d.day)
                    timeline.append((time, summary))
                    l += 1


        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)

    def load(self, ignored_topics):
        key = ' '.join(sorted(ignored_topics))
        if self.date_ranker:
            if self.key_to_model:
                self.date_ranker.model = self.key_to_model[key]

    def train(self, collection, topic):
        pass

