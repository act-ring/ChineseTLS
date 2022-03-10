"""
@author: Li Xi
@file: prepare_dataset_cn.py
@time: 2020/10/15 11:20
@desc:
"""

import json
import os
import uuid  # create unique i

from pyltp import SentenceSplitter, Segmentor


def write_jsonl(items, path, batch_size=100, override=True):
    if override:
        with open(path, 'w'):
            pass

    batch = []
    for i, x in enumerate(items):
        if i > 0 and i % batch_size == 0:
            with open(path, 'a') as f:
                output = '\n'.join(batch) + '\n'
                f.write(output)
            batch = []
        raw = json.dumps(x, ensure_ascii=False)
        batch.append(raw)

    if batch:
        with open(path, 'a') as f:
            output = '\n'.join(batch) + '\n'
            f.write(output)


segmentor = Segmentor()
segmentor.load(os.path.join('/home/LAB/lixi/ltp_data_v3.4.0', 'cws.model'))

news_dir = "/home/LAB/lixi/resource/tlcn/TLCN/data/"
output_dir = '../datasets/TLCN2/'


topics = os.listdir(news_dir)


for topic in topics:
    if topic[0] == '.': continue
    print(topic)
    topic_dir = output_dir + topic
    if not os.path.exists(topic_dir):
        os.makedirs(topic_dir)
    article_output = os.path.join(topic_dir, 'articles.jsonl')
    timeline_output = os.path.join(topic_dir, 'timelines.jsonl')
    keywords_output = os.path.join(topic_dir, 'keywords.json')
    article_dir = os.path.join(news_dir, topic, "news")
    timeline_input = os.path.join(news_dir, topic, "timeline.txt")

    # 处理articles
    articles = os.listdir(article_dir)
    output_articles = []
    for article in articles:
        if article[0] == '.': continue
        article_path = os.path.join(article_dir, article)
        t = article[:10]
        with open(article_path, 'r', encoding='utf-8') as f:
            article_raw = json.loads(f.read())


        for item in article_raw:
            tmp_article = {
                    'id': str(uuid.uuid1()),
                    'url': item['url'],
                    'time': t,
                    'title': item['title'],
                    'text': item['content']
                }
            output_articles.append(tmp_article)
    write_jsonl(output_articles, article_output)

    # 处理timelines
    with open(timeline_input, 'r', encoding='utf-8') as f:
        content = f.readlines()
        content = [x.strip('\n').strip() for x in content]
    times = []
    summarys = []
    for i in range(2, len(content), 3):
        times.append(content[i])
        summarys.append(content[i + 2])
    ret = {}
    for t, s in zip(times, summarys):
        sentence = SentenceSplitter.split(s)
        tokens = []
        for item in sentence:
            ts = segmentor.segment(item)
            ts = [x for x in ts]
            tokens.append(" ".join(ts))
        ret[t] = tokens

    ret = sorted(ret.items(), key=lambda x: x[0])
    write_jsonl([ret], timeline_output)

    # 处理keywords
    keywords = content[0].split()
    with open(keywords_output, 'w', encoding='utf-8') as f:
        f.write(json.dumps(keywords, ensure_ascii=False))



print('done')
