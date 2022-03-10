"""
@author: Li Xi
@file: data.py
@time: 2020/10/12 14:56
@desc:
"""
import os
import pathlib

from . import utils


class Sentence:
    def __init__(self, text, token, pub_time, time, name_entity_num, is_title=False):
        self.text = text  # str
        self.token = token  # str (with space)
        self.token_list = token.split()  # str list
        self.pub_time = pub_time  # datetime
        self.time = time  # datetime list
        self.is_title = is_title  # bool
        self.name_entity_num = name_entity_num  # int

    def to_dict(self):
        return {
            'text': self.text,
            'token': self.token,
            'token_list': self.token_list,
            'time': self.time,
            'pub_time': self.pub_time,
            'name_entity_num': self.name_entity_num
        }

    def get_time(self):
        return self.time + [self.pub_time]

    def get_date(self):
        if len(self.time) == 0:
            return None
        else:
            return sorted(self.time)[0]


class Article:
    def __init__(self, title, tokenized_title, text, tokenized_text, time, id, sentences=None, title_sentence=None):
        self.title = title  # str
        self.tokenized_title = tokenized_title  # str (with space)
        self.text = text  # str
        self.tokenized_text = tokenized_text  # str (with space)
        self.time = time  # datetime
        self.id = id  # str
        self.sentences = sentences  # dict list
        self.title_sentence = title_sentence  # dict

    def to_dict(self):
        return {
            'title': self.title,
            'tokenized_title': self.tokenized_title,
            'text': self.text,
            'tokenized_text': self.tokenized_text,
            'time': self.time,
            'id': self.id,
            'sentences': [s.to_dict() for s in self.sentences],
            'title_sentence': self.title_sentence.to_dict() if self.title_sentence else None
        }


class Dataset:
    def __init__(self, path):
        self.path = pathlib.Path(path)  # path
        self.topics = self._get_topics()  # str list
        self.collections = self._load_collections()  # Article list

    def _get_topics(self):
        return sorted(os.listdir(self.path))

    def _load_collections(self):
        collections = []
        for topic in self.topics:
            topic_path = self.path / topic
            c = ArticleCollection(topic_path)
            collections.append(c)
        return collections


class ArticleCollection:
    def __init__(self, path, start=None, end=None):
        self.name = os.path.basename(path)  # str
        self.path = pathlib.Path(path)  # path
        self.start = start  # datetime
        self.end = end  # datetime
        self.timelines = self._load_timelines()  # Timeline list

    def _load_timelines(self):
        timelines = []
        path = self.path / 'timelines.jsonl'
        if not path.exists():
            return []
        for raw_tl in utils.read_jsonl(path):
            if raw_tl:
                tl = Timeline(raw_tl)
                timelines.append(tl)
                tmp_start, tmp_end = tl.times[0], tl.times[-1]
                self.start = tmp_start if self.start is None or self.start > tmp_start else self.start
                self.end = tmp_end if self.end is None or self.end < tmp_end else self.end
        self.start = utils.str_to_date(self.start) if self.start else None
        self.end = utils.str_to_date(self.end) if self.end else None
        return timelines

    def articles(self):
        path1 = self.path / 'articles.preprocess.jsonl'
        path2 = self.path / 'articles.preprocess.jsonl.gz'
        if path1.exists():
            articles = utils.read_jsonl(path1)
        else:
            articles = utils.read_jsonl_gz(path2)
        for a_ in articles:
            a = load_article(a_)
            if self.start and a.time < self.start:
                continue
            if self.end and a.time > self.end:
                continue
            yield a

    def time_batches(self):
        path1 = self.path / 'articles.preprocess.jsonl'
        path2 = self.path / 'articles.preprocess.jsonl.gz'
        if path1.exists():
            articles = utils.read_jsonl(path1)
            articles = utils.read_jsonl_gz(path2)
        time = None
        batch = []
        for a_ in articles:
            a = load_article(a_)
            a_time = a.time

            if self.start and a_time < self.start:
                continue

            if self.end and a_time > self.end:
                break

            if time and a_time > time:
                yield time, batch
                time = a_time
                batch = [a]
            else:
                batch.append(a)
                time = a_time
        yield time, batch

    def times(self):
        path1 = self.path / 'articles.preprocess.jsonl'
        path2 = self.path / 'articles.preprocess.jsonl.gz'
        if path1.exists():
            articles = utils.read_jsonl(path1)
        else:
            articles = utils.read_jsonl_gz(path2)
        times = []
        for a in articles:
            times.append(a['time'])
        return times


class Timeline:
    def __init__(self, items):
        self.items = sorted(items, key=lambda x: x[0])  # list [str, list]
        self.items = [[utils.str_to_date(t), s] for t, s in self.items]  # list [datetime, list]
        self.time_to_summaries = dict((t, s) for t, s in self.items)  # dict
        self.date_to_summaries = dict((t.date(), s) for t, s in self.items)  # dict
        self.times = sorted(self.time_to_summaries)  # str list

    def __getitem__(self, item):
        return self.time_to_summaries[item]

    def __len__(self):
        return len(self.items)

    def __str__(self):
        lines = []
        for t, summary_list in self.items:
            lines.append('[{}]'.format(t.date()))
            for summary in summary_list:
                for sent in summary:
                    lines.append(sent)
                lines.append('-' * 50)
            return '\n'.join(lines)

    def to_dict(self):
        items = dict([(str(t), s) for (t, s) in self.items])
        return items


def load_dataset(path):
    dataset = Dataset(path)
    return dataset


def load_article(article_dict):
    sentences = list(map(load_sentence, article_dict['sentences']))
    if article_dict.get('title'):
        title_sentence = load_sentence({
            'text': article_dict['title'],
            'token': article_dict['tokenized_title'],
            'time': [],
            'pub_time': article_dict['time'],
            'name_entity_num': article_dict['title_entity_num']
        })
        title_sentence.is_title = True
    else:
        title_sentence = None
    article_time = utils.str_to_date(article_dict['time'])

    return Article(
        title=article_dict['title'],
        tokenized_title=article_dict['tokenized_title'],
        text=article_dict['text'],
        tokenized_text=article_dict['tokenized_text'],
        time=article_time,
        id=article_dict['id'],
        sentences=sentences,
        title_sentence=title_sentence
    )


def load_sentence(sent_dict):
    return Sentence(
        text=sent_dict['text'],
        token=sent_dict['token'],
        pub_time=utils.str_to_date(sent_dict['pub_time']),
        time=list(map(utils.str_to_date, sent_dict['time'])),
        name_entity_num=sent_dict['name_entity_num']
    )


def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [x.strip('\n').strip() for x in stopwords]
        stopwords = [x for x in stopwords if len(x) > 0]
    return stopwords


def get_average_summary_length(ref_tl):
    lens = []
    for date, summary in ref_tl.dates_to_summaries.items():
        lens.append(len(summary))
    k = sum(lens) / len(lens)
    return round(k)
