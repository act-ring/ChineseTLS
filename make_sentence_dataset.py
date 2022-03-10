import sys
sys.setrecursionlimit(100000)
import argparse
from pathlib import Path
from pprint import pprint

from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.evaluation import rouge
import json
from tls import clust, datewise, data, utils, summarizers, chieu, asr, oracle
import random



def main(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    print(dataset_path)
    with open("test/tlcn.oracle.text.doc2vec.more.json", "r", encoding='utf-8') as f:
        content = json.loads(f.read())
    ground_timelines = []
    negative_sentences = []
    pred_timelines = content["results"]
    for i, collection in enumerate(dataset.collections):
        # print(i)
        ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]
        # print(len(ref_timelines))
        # exit(0)
        for j, ref_timeline in enumerate(ref_timelines):
            # print("j", j)
            ground_timelines.append(ref_timeline)

        # get negative sentences
        # pub time && token

        ground_date = []
        for g in ground_timelines: ground_date += list(g.dates_to_summaries.keys())
        ground_date_str = [str(x) for x in ground_date]
        articles = collection.articles()
        # print(len(list(articles)))
        for article in articles:
            article_date = article.time.date()
            tmp_negative = []
            if article_date not in ground_date_str:
                for sent in article.sentences:
                    if len(sent.token.split()) >= 10:
                        tmp_negative.append(sent.token)
            random.shuffle(tmp_negative)
            negative_sentences+=tmp_negative[:10]

    random.shuffle(negative_sentences)
    negative_sentences = negative_sentences[:5000]

    print(len(ground_timelines))
    print(len(pred_timelines))
    print(len(negative_sentences))

    output = []
    for g, p in zip(ground_timelines, pred_timelines):
        for gtime in list(g.dates_to_summaries.keys()):
            g_key = str(gtime) + " 00:00:00"
            # print(g_key)
            # print(p.keys())
            if g_key in p[2]:
                if len( p[2][g_key])<=1:
                    continue
                output.append({
                    "ref": g.dates_to_summaries[gtime],
                    "pred": p[2][g_key][:len(p[2][g_key])//3+1],
                    "label": 1
                })

                output.append({
                    "ref": g.dates_to_summaries[gtime],
                    "pred": p[2][g_key][0-(len(p[2][g_key]) // 3):],
                    "label": 2
                })

                output.append({
                    "ref": g.dates_to_summaries[gtime],
                    "pred": random.sample(negative_sentences, len(g.dates_to_summaries[gtime])), # 构造负例，句子数量和ref句子数量相等
                    "label": 0
                })
    print(len(output))

    with open("present_dataset/dataset_all_more.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(output, ensure_ascii=False))
    # 385 positive
    # 770 all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="datasets/TLCN/")

    main(parser.parse_args())
