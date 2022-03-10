"""
@author: Li Xi
@file: train_regression.py
@time: 2020/10/19 20:35
@desc:
"""
import pickle
from tls import regression, data

system = regression.Regression(
            language="chinese",
            mode="eval"
        )
dataset_path = "datasets/TLCN"
# dataset_path = "datasets/english-example"
dataset = data.Dataset(dataset_path)
ret_models = {}
for t , collection in zip(dataset.topics, dataset.collections):
    system.train(dataset, t)
    ret_models[t] = system.model
    print(t)
dataset_name = "TLCN"
# dataset_name = "english-example"
models_path = "resources/supervised_regression.{}-1.pkl".format(dataset_name)
pickle.dump(ret_models, open(models_path, 'wb'))

print('done')
