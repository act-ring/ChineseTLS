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

import codecs
import datetime
import gzip
import io
import json
import os
import pickle
import shutil
import tarfile

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def force_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def dict_to_dense_vector(d, key_to_idx):
    x = np.zeros(len(key_to_idx))
    for key, i in key_to_idx.items():
        x[i] = d[key]
    return x


def read_file(path):
    with codecs.open(path, 'r',  errors='ignore') as f:#encoding='utf-8',
        text = f.read()
    return text


def write_file(s, path):
    with open(path, 'w') as f:
        f.write(s)


def read_json(path):
    text = read_file(path)
    return json.loads(text)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            # try:
            #     json.loads(line)
            # except:
            #     print(path)
            yield json.loads(line)


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


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def write_gzip(text, path):
    with gzip.open(path, 'wb') as output:
        with io.TextIOWrapper(output) as enc:#, encoding='utf-8'
            enc.write(text)


def read_gzip(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            content = dec.read()
    return content


def read_jsonl_gz(path):
    with gzip.open(path, 'rb') as input_file:
        with io.TextIOWrapper(input_file) as dec:
            for line in dec:
                yield json.loads(line)


def read_tar_gz(path):
    contents = []
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            content = f.read()
            contents.append(content)
    return contents


def read_json_tar_gz(path):
    contents = read_tar_gz(path)
    raw_data = contents[0]
    return json.loads(raw_data, strict=False)


def get_date_range(start, end):
    diff = end - start
    date_range = []
    for n in range(diff.days + 1):
        t = start + datetime.timedelta(days=n)
        date_range.append(t)
    return date_range


def days_between(t1, t2):
    return abs((t1 - t2).days)


def any_in(items, target_list):
    return any([item in target_list for item in items])


def csr_item_generator(M):
    """Generates tuples (i,j,x) of sparse matrix."""
    for row in range(len(M.indptr) - 1):
        i, j = M.indptr[row], M.indptr[row + 1]
        for k in range(i, j):
            yield (row, M.indices[k], M.data[k])


def max_normalize_matrix(A):
    try:
        max_ = max(A.data)
        for i, j, x in csr_item_generator(A):
            A[i, j] = x / max_
    except:
        pass
    return A


def gzip_file(inpath, outpath, delete_old=False):
    with open(inpath, 'rb') as infile:
        with gzip.open(outpath, 'wb') as outfile:
            outfile.writelines(infile)
    if delete_old:
        os.remove(inpath)


def normalise(X, method='standard'):
    if method == 'max':
        return X / X.max(0)
    elif method == 'minmax':
        return MinMaxScaler().fit_transform(X)
    elif method == 'standard':
        return StandardScaler().fit_transform(X)
    elif method == 'robust':
        return RobustScaler().fit_transform(X)
    else:
        raise ValueError('normalisation method not known: {}'.format(method))


def normalize_vectors(vector_batches, mode='standard'):
    if mode == 'max':
        normalize = lambda X: X / X.max(0)
    elif mode == 'minmax':
        normalize = lambda X: MinMaxScaler().fit_transform(X)
    elif mode == 'standard':
        normalize = lambda X: StandardScaler().fit_transform(X)
    elif mode == 'robust':
        normalize = lambda X: RobustScaler().fit_transform(X)
    else:
        normalize = lambda X: X
    norm_vectors = []
    for vectors in vector_batches:
        X = np.array(vectors)
        X_norm = normalize(X)
        norm_vectors += list(X_norm)
    return norm_vectors


def strip_to_date(t):
    return datetime.datetime(t.year, t.month, t.day)


def print_tl(tl):
    for t, sents in tl.items:
        print('[{}]'.format(t.date()))
        for s in sents:
            print(' '.join(s.split()))
        print('---')


def str_to_date(t_str):
    if type(t_str) is str:
        year, month, day = t_str[:10].split('-')
        return datetime.datetime(int(year), int(month), int(day))
    return t_str
