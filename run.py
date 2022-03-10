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
import argparse
from pathlib import Path

from tls import clust, datewise, data, utils, summarizers


def run(tls_model, dataset, outpath):
    n_topics = len(dataset.collections)
    outputs = []

    for i, collection in enumerate(dataset.collections):
        topic = collection.name
        times = [a.time for a in collection.articles()]
        # setting start, end, L, K manually instead of from ground-truth
        collection.start = min(times)
        collection.end = max(times)
        l = 8  # timeline length (dates)
        k = 1  # number of sentences in each summary

        timeline = tls_model.predict(
            collection,
            max_dates=l,
            max_summary_sents=k,

        )

        print('*** TIMELINE ***')
        utils.print_tl(timeline)

        outputs.append(timeline.to_dict())

    if outpath:
        utils.write_json(outputs, outpath)


def main(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    if args.method == 'datewise':
        # load regression models for date ranking
        key_to_model = utils.load_pkl(args.model)
        models = list(key_to_model.values())
        date_ranker = datewise.SupervisedDateRanker(method='regression')
        # there are multiple models (for cross-validation),
        # we just an arbitrary model, the first one
        date_ranker.model = models[0]
        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=2, pub_end=2)
        summarizer = summarizers.CentroidOpt()
        system = datewise.DatewiseTimelineGenerator(
            language=args.language,
            mode=args.mode,
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model=key_to_model
        )
    elif args.method == 'clust':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer()
        summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            language=args.language,
            mode=args.mode,
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=2,
            unique_dates=True,
        )
    else:
        raise ValueError(f'Method not found: {args.method}')

    run(system, dataset, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--language', required=True)
    parser.add_argument('--mode', default="predict")
    parser.add_argument('--method', required=True)
    parser.add_argument('--model', default=None,
                        help='model for date ranker')
    parser.add_argument('--output', default=None)
    main(parser.parse_args())
    # python run_without_eval.py --dataset datasets/chinese-example/ --method clust --output outputs/timline.chinese.eval.jsonl --language chinese --mode eval
    # python run_without_eval.py --dataset datasets/english-example/ --method clust --output outputs/timline.english.eval.jsonl --language english --mode eval
    # python run_without_eval.py --dataset datasets/chinese-example/ --method clust --output outputs/timline.chinese.jsonl --language chinese
    # python run_without_eval.py --dataset datasets/english-example/ --method clust --output outputs/timline.english.jsonl --language english
    # python run_without_eval.py --dataset datasets/english-example/ --method datewise --output outputs/timline.english.datewise.jsonl --language english --model /home/LAB/lixi/projects/news-tls/resources/datewise/supervised_date_ranker.t17.pkl
    # python run_without_eval.py --dataset datasets/chinese-example/ --method datewise --output outputs/timline.chinese.datewise.jsonl --language chinese --model /home/LAB/lixi/projects/news-tls/resources/datewise/supervised_date_ranker.t17.pkl
