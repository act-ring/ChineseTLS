"""
@author: Li Xi
@file: prepare_dataset_t17.py
@time: 2020/10/22 20:48
@desc:
"""
import json
import os

import nltk

dataset_dir = "/home/LAB/lixi/resource/t17/Data/"
data_files = os.listdir(dataset_dir)
output_dir = '/home/LAB/lixi/projects/news-tls/datasets/t17/'

print('start.....')

def write_jsonl(items, path, batch_size=100, override=False):
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

for topic in data_files:
    if topic[0] == ".":
        continue
    output_topic = topic.split('_')[0]
    print(topic)
    topic_dir = output_dir + output_topic
    if not os.path.exists(topic_dir):
        os.makedirs(topic_dir)
    article_output = os.path.join(topic_dir, 'articles.jsonl')
    timeline_output = os.path.join(topic_dir, 'timelines.jsonl')

    article_input_dir = os.path.join(dataset_dir, topic,  "InputDocs/")
    date_articles_dir = os.listdir(article_input_dir)

    output_articles = []
    for da in date_articles_dir:
        if da[0] == '.':
            continue
        arts_da = os.path.join(article_input_dir , da)
        art_files = os.listdir(arts_da)

        for af in art_files:
            if af[0] == '.':
                continue
            fname = os.path.join(arts_da, af)
            with open(fname, 'r', encoding='utf-8') as f:
                content = f.read()

            id = af.split('.')[0]
            url = ""
            ti = da
            title = ""
            text = content

            output_articles.append({
                'id': af.split('.')[0],
                'url': "",
                'time': ti,
                'title': "",
                'text': content
            })
    write_jsonl(output_articles, article_output)


    timeline_input_dir =os.path.join(dataset_dir, topic, 'timelines')
    timeline_files = os.listdir(timeline_input_dir)

    ret_timelines = []
    for tf in timeline_files:
        if tf[0] == '.':
            continue
        tfpath = os.path.join(timeline_input_dir, tf)
        with open(tfpath, 'r', encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip('\n').strip() for x in content]
        times = []
        summarys = []
        summ = ""
        for i in range(len(content)):
            if len(content[i]) == 10:
                times.append(content[i])
            elif content[i] == '--------------------------------':
                summarys.append(summ)
                summ = ""
            else:
                summ += content[i]

        ret = {}
        for t, s in zip(times, summarys):
            sentence = nltk.sent_tokenize(s)
            tokens = []
            for item in sentence:
                ts = nltk.word_tokenize(item)
                ts = [x for x in ts]
                tokens.append(" ".join(ts))
            ret[t] = tokens
        ret = sorted(ret.items(), key=lambda x: x[0])
        ret_timelines.append(ret)
    write_jsonl(ret_timelines, timeline_output)

print('done')
