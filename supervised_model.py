"""
@author: Li Xi
@file: supervised_model.py
@time: 2020/10/19 10:33
@desc:
"""
import pickle

from tls import data
from tls.datewise import SupervisedDateRanker
from tilse.data.timelines import Timeline as TilseTimeline
import numpy as np
from sklearn.linear_model import LinearRegression

# dataset_path = "datasets/chinese-example/"
dataset_path = "datasets/TLCN/"
ranker = SupervisedDateRanker()
dataset = data.Dataset(dataset_path)

ret_model = {}
topic_list = []
total_dates = []
X = []
Y = []
for i, collection in enumerate(dataset.collections):
    dates, x = ranker.extract_features(collection)
    ref_timelines = [TilseTimeline(tl.date_to_summaries)
                     for tl in collection.timelines]
    timeline_dates = []
    for tl in ref_timelines:
        timeline_dates += tl.dates_to_summaries.keys()
    timeline_dates  = sorted(list(set(timeline_dates)))
    y = [1 if d in timeline_dates else 0for d in dates]
    total_dates += dates
    for item in x:
        X.append(item)
    for item in y:
        Y.append(item)
    x = np.array(x)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    reg.score(x, y)
    ret_model[collection.name] = {}
    ret_model[collection.name]['model'] = reg
    ret_model[collection.name]['y'] = y
    ret_model[collection.name]['X'] = x
    print(i)


X = np.array(X)
Y = np.array(Y)
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)

ret_model[" ".join(sorted(topic_list))] = {}
ret_model[" ".join(sorted(topic_list))]['model'] = reg
ret_model[" ".join(sorted(topic_list))]['X'] = X
ret_model[" ".join(sorted(topic_list))]['y'] = Y

dataset_name = "TLCN"
models_path = "resources/datewise/supervised_date_ranker.{}-1.pkl".format(dataset_name)
pickle.dump(ret_model, open(models_path, 'wb'))



# loaded_model = pickle.load(open(models_path, 'rb'))
# result = loaded_model.predict(np.array([[1,1,0,1,1,0,0]]))
# print(result)
