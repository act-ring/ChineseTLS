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
import sys
sys.setrecursionlimit(100000)
import argparse
from pathlib import Path
from pprint import pprint

from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.evaluation import rouge

from tls import clust, datewise, data, utils, summarizers, chieu, asr, oracle, vectorizors, submodular


def get_scores(metric_desc, pred_tl, groundtruth, evaluator):
    if metric_desc == "concat":
        return evaluator.evaluate_concat(pred_tl, groundtruth)
    elif metric_desc == "agreement":
        return evaluator.evaluate_agreement(pred_tl, groundtruth)
    elif metric_desc == "align_date_costs":
        return evaluator.evaluate_align_date_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs":
        return evaluator.evaluate_align_date_content_costs(
            pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs_many_to_one":
        return evaluator.evaluate_align_date_content_costs_many_to_one(
            pred_tl, groundtruth)


def zero_scores():
    return {'f_score': 0., 'precision': 0., 'recall': 0.}


def evaluate_dates(pred, ground_truth):
    pred_dates = pred.get_dates()
    ref_dates = ground_truth.get_dates()
    shared = pred_dates.intersection(ref_dates)
    n_shared = len(shared)
    n_pred = len(pred_dates)
    n_ref = len(ref_dates)
    prec = n_shared / n_pred
    rec = n_shared / n_ref
    if prec + rec == 0:
        f_score = 0
    else:
        f_score = 2 * prec * rec / (prec + rec)
    return {
        'precision': prec,
        'recall': rec,
        'f_score': f_score,
    }


def get_average_results(tmp_results):
    rouge_1 = zero_scores()
    rouge_2 = zero_scores()
    date_prf = zero_scores()
    for rouge_res, date_res, _ in tmp_results:
        metrics = [m for m in date_res.keys() if m != 'f_score']
        for m in metrics:
            rouge_1[m] += rouge_res['rouge_1'][m]
            rouge_2[m] += rouge_res['rouge_2'][m]
            date_prf[m] += date_res[m]
    n = len(tmp_results)
    for result in [rouge_1, rouge_2, date_prf]:
        for k in ['precision', 'recall']:
            result[k] = result[k] / n if n != 0 else 0.0
        prec = result['precision']
        rec = result['recall']
        if prec + rec == 0:
            result['f_score'] = 0.
        else:
            result['f_score'] = (2 * prec * rec) / (prec + rec)
    return rouge_1, rouge_2, date_prf


def evaluate(tls_model, dataset, result_path, trunc_timelines=False, time_span_extension=0):
    yy, xx = [], []
    results = []
    metric = 'align_date_content_costs_many_to_one'
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    n_topics = len(dataset.collections)
    knee = []

    for i, collection in enumerate(dataset.collections):

        if i < 63:
            continue

        ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]
        topic = collection.name
        n_ref = len(ref_timelines)

        # tls_model.train(dataset, collection.name)

        # if trunc_timelines:
        #     ref_timelines = data.truncate_timelines(ref_timelines, collection)

        for j, ref_timeline in enumerate(ref_timelines):

            print(f'topic {i + 1}/{n_topics}: {topic}, ref timeline {j + 1}/{n_ref}')
            try:
                tls_model.load(ignored_topics=[collection.name])
            except:
                # print("rrrrrrrrr")
                continue
            # continue
            ref_dates = sorted(ref_timeline.dates_to_summaries)

            # start, end = data.get_input_time_span(ref_dates, time_span_extension)
            #
            # collection.start = start
            # collection.end = end

            # utils.plot_date_stats(collection, ref_dates)

            l = len(ref_dates)

            # k = data.get_average_summary_length(ref_timeline)
            # pred_timeline_ = tls_model.predict(
            #     collection,
            #     max_dates=l,
            #     max_summary_sents=k,
            #     ref_tl=ref_timeline  # only oracles need this
            # )
            try:
                k = data.get_average_summary_length(ref_timeline)
                pred_timeline_ = tls_model.predict(
                    collection,
                    max_dates=l,
                    max_summary_sents=k,
                    ref_tl=ref_timeline  # only oracles need this
                )
            except Exception as e:
                print(e)
                continue
            # k = data.get_average_summary_length(ref_timeline)
            # pred_timeline_ = tls_model.predict(
            #     collection,
            #     max_dates=l,
            #     max_summary_sents=k,
            #     ref_tl=ref_timeline  # only oracles need this
            # )

            print('*** PREDICTED ***')
            # utils.print_tl(pred_timeline_)
            # if len(pred_timeline_.date_to_summaries) == 0:
            #     continue

            print('timeline done')


            pred_timeline = TilseTimeline(pred_timeline_.date_to_summaries)
            sys_len = len(pred_timeline.get_dates())
            ground_truth = TilseGroundTruth([ref_timeline])

            try:
                rouge_scores = get_scores(
                    metric, pred_timeline, ground_truth, evaluator)
                date_scores = evaluate_dates(pred_timeline, ground_truth)
            except:
                continue

            print('sys-len:', sys_len, 'gold-len:', l, 'gold-k:', k)

            print('Alignment-based ROUGE:')
            pprint(rouge_scores)
            print('Date selection:')
            pprint(date_scores)
            print('-' * 100)
            # results.append((rouge_scores, date_scores, pred_timeline_.to_dict()))
            try:
                knee_info = tls_model.ranked
                knee_info['ref_l'] = int(str(l))
            #     tmp = (rouge_scores, date_scores, pred_timeline_.to_dict(),
            #            knee_info)
                knee.append(knee_info)
            except:
                print('not knee')
            tmp = (rouge_scores, date_scores, pred_timeline_.to_dict())
            results.append(tmp)

    avg_results = get_average_results(results)
    print('Average results:')
    pprint(avg_results)
    output = {
        'average': avg_results,
        'results': results,
        'knee': knee

    }
    utils.write_json(output, result_path)
    # print(xx)
    # print(yy)


def main(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    if args.vectorizer == 'doc2vec':
        vectorizer = vectorizors.Doc2vecVectorizer(language=args.language)
    elif args.vectorizer == 'sent-bert':
        vectorizer = vectorizors.SentBertVectorizer()
    else:
        vectorizer = None

    if args.method == 'datewise':
        resources = Path(args.resources)
        models_path = resources / 'supervised_date_ranker.{}.pkl'.format(
            dataset_name
        )
        # load regression models for date ranking
        key_to_model = utils.load_pkl(models_path)
        if args.date_select == "mention":
            date_ranker = datewise.MentionCountDateRanker()
        elif args.date_select == "regression":
            date_ranker = datewise.SupervisedDateRanker(method='regression')
        elif args.date_select == "multi-scale":
            date_ranker = datewise.MultiScaleDateRanker(method='regression')
            # date_ranker = datewise.MultiScaleDateRanker(args.alpha, args.beta)
        else:
            raise ValueError(f'date_select not found: {args.date_select}')

        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=5, pub_end=2)
        if args.summarizer == "textrank":
            summarizer = summarizers.TextRank()
        elif args.summarizer == "centriodrank":
            summarizer = summarizers.CentroidRank()
        else:
            summarizer = summarizers.CentroidOpt()
        system = datewise.DatewiseTimelineGenerator(
            language=args.language,
            mode=args.mode,
            vectorizer=vectorizer,
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model=key_to_model,
            l=args.l,
            k=args.k
        )
    elif args.method == 'clust':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer()
        if args.summarizer == "textrank":
            summarizer = summarizers.TextRank()
        elif args.summarizer == "centriodrank":
            summarizer = summarizers.CentroidRank()
        else:
            summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            language=args.language,
            mode=args.mode,
            vectorizer=vectorizer,
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=5,
            unique_dates=True,
            l=args.l,
            k=args.k
        )
    elif args.method == 'chieu':
        system = chieu.Chieu(
            language=args.language,
            mode=args.mode,
            vectorizer=vectorizer,
            n=args.n
        )
    elif args.method == 'submodular':
        system = submodular.Submodular(
            language=args.language,
            mode=args.mode,
            vectorizer=vectorizer
            # n=args.n
        )
    elif args.method == 'asr':
        system = asr.AsrTimelineGenerator(
            language=args.language,
            mode=args.mode,
            vectorizer=vectorizer,
            date_ranker=None,
            summarizer=None,
            sent_collector=None,
            key_to_model=None,
            l=args.l,
            k=args.k
        )
    elif args.method == 'oracle':
        resources = Path(args.resources)
        models_path = resources / 'supervised_date_ranker.{}.pkl'.format(
            dataset_name
        )
        # load regression models for date ranking
        key_to_model = utils.load_pkl(models_path)
        # print(key_to_model.keys())
        if args.oracle_type == "text":
            date_ranker = datewise.SupervisedDateRanker(method='regression')
        else:
            date_ranker = None

        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=5, pub_end=2)
        if args.oracle_type == "full" or args.oracle_type == "text":
            summarizer = summarizers.OracleSummarizer()
        else:
            summarizer = summarizers.CentroidOpt()
        system = oracle.OracleTimelineGenerator(
            language=args.language,
            oracle_type=args.oracle_type,
            mode=args.mode,
            vectorizer=vectorizer,
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model=key_to_model,
            l=args.l,
            k=args.k
        )
    else:
        raise ValueError(f'Method not found: {args.method}')

    if dataset_name == 'entities':
        evaluate(system, dataset, args.output, trunc_timelines=True, time_span_extension=7)
    else:
        evaluate(system, dataset, args.output, trunc_timelines=False, time_span_extension=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--vectorizer', default="doc2vec")
    parser.add_argument('--language', required=True)
    parser.add_argument('--mode', default="predict")
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--k', action='store_true', default=False)
    parser.add_argument('--l', action='store_true', default=False)
    parser.add_argument('--n', action='store_true', default=False)
    parser.add_argument('--summarizer', default=None)
    parser.add_argument('--oracle_type', default=None)
    parser.add_argument('--method', required=True)
    parser.add_argument('--resources', default=None,
                        help='model resources for tested method')
    parser.add_argument('--output', default=None)
    parser.add_argument('--date_select', default=None)


    main(parser.parse_args())

    # python evaluate.py  --dataset datasets/TLCN/ --method clust --output outputs/timline.chinese.eval.jsonl --language chinese --mode eval --summarizer centriodrank --resources resources/datewise --date_select mention --k --l
    # python evaluate.py --dataset datasets/english-example/ --method clust --output outputs/timline.english.eval.jsonl --language english --mode eval

    # python evaluate.py  --dataset datasets/TLCN/ --method clust --output outputs/timline.TLCN.eval.clust.jsonl --language chinese --mode eval
    # python evaluate.py  --dataset datasets/TLCN/ --method datewise --output outputs/timline.TLCN.eval.datewise.mention.jsonl --language chinese --mode eval --resources resources/datewise  --date_select mention
    # python evaluate.py  --dataset datasets/TLCN/ --method chieu --output outputs/timline.TLCN.eval.chieu.jsonl --language chinese --mode eval
    # python evaluate.py  --dataset datasets/TLCN/ --method submodular --output outputs/timline.TLCN.eval.submodular.jsonl --language chinese --mode eval
    # python evaluate.py  --dataset datasets/TLCN/ --method regression --output outputs/timline.TLCN.eval.regression.jsonl --language chinese --mode eval --resources resources
    # python evaluate.py  --dataset datasets/english-example/ --method regression -output outputs/timline.english-example.eval.regression.jsonl --language english --mode eval --resources resources
    # python evaluate.py  --dataset datasets/chinese-example/ --method regression --output outputs/timline.chinese-example.eval.regression.jsonl --language chinese --mode eval --resources resources
    # python evaluate.py  --dataset datasets/crisis/ --method adaptive_clust --output outputs/timline.crisis.eval.adaptive_clust.jsonl --language english --mode eval --resources resources
    # python evaluate.py  --dataset datasets/crisis/ --method datewise --output outputs/timline.crisis.eval.datewise.jsonl --language english --mode eval --resources resources/datewise  --date_select mention
    # python evaluate.py  --dataset datasets/TLCN/ --method adaptive_clust --output outputs/timline.TLCN.eval.adaptive_clust.jsonl --language chinese --mode eval --resources resources/  --date_select mention


    # python evaluate.py  --dataset datasets/TLCN/ --method adaptive_datewise --output outputs/timline.TLCN.eval.adaptive_datewise.mention.jsonl --language chinese --mode eval --resources resources/datewise  --date_select mention
    # python evaluate.py  --dataset datasets/TLCN/ --method adaptive_datewise --output outputs/timline.TLCN.eval.adaptive_datewise.regression.jsonl --language chinese --mode eval --resources resources/datewise  --date_select regression

    # python evaluate.py --dataset datasets/t17/ --method clust --output outputs/timline.t17.eval.clust.jsonl --language english --mode eval


    # python evaluate.py  --dataset datasets/TLCN/ --method adaptive --output outputs/timline.TLCN.eval.adaptive.regresssion.jsonl --language chinese --mode eval --resources resources/datewise  --date_select regression

