"""
@author: Li Xi
@file: preprocess.py
@time: 2020/10/12 15:50
@desc:
"""
import argparse
import datetime
import os
import pathlib

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from pyltp import SentenceSplitter, Segmentor, Postagger, NamedEntityRecognizer
from recognizers_date_time import recognize_datetime, Culture

from tls import utils

# init ltp (use in chinese preprocess)
segmentor = Segmentor()
segmentor.load(os.path.join('/home/LAB/lixi/ltp_data_v3.4.0', 'cws.model'))
pos = Postagger()
pos.load(os.path.join('/home/LAB/lixi/ltp_data_v3.4.0', 'pos.model'))
ner = NamedEntityRecognizer()
ner.load(os.path.join('/home/LAB/lixi/ltp_data_v3.4.0', 'ner.model'))


def chinese_tokenizer(text, title):
    sentences = SentenceSplitter.split(text)
    tokenized_doc = ''
    tokenized_sent = []
    for sent in sentences:
        tokens = segmentor.segment(sent)
        tokens = [x for x in tokens]
        tmp_sentence = ' '.join(tokens)
        tokenized_doc += tmp_sentence + '\n'
        tokenized_sent.append(tmp_sentence.strip())
    tokenized_title = segmentor.segment(title)
    tokenized_title = " ".join([x for x in tokenized_title])
    return tokenized_doc, tokenized_sent, tokenized_title, sentences


def english_tokenizer(text, title):
    sentences = sent_tokenize(text)
    tokenized_doc = ''
    tokenized_sent = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tmp_sentence = ' '.join(tokens)
        tokenized_doc += tmp_sentence + '\n'
        tokenized_sent.append(tmp_sentence.strip())
    tokenized_title = word_tokenize(title)
    tokenized_title = " ".join(tokenized_title)
    return tokenized_doc, tokenized_sent, tokenized_title, sentences


def english_name_entity_counter(text):
    words = text.split()
    tagged_sentence = pos_tag(words)
    ne_chunked_sent = ne_chunk(tagged_sentence)
    named_entities = []
    for tagged_tree in ne_chunked_sent:
        # extract only chunks having NE labels
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  # get NE name
            named_entities.append(entity_name)
            # get unique named entities
    return len(named_entities)


def chinese_name_entity_counter(text):
    words = text.split()
    postags = pos.postag(words)
    ne_tags = ner.recognize(words, postags)
    sentence_len = len(words)
    entity_types = ['Nh', 'Ni', 'Nl']
    ret_entity = {'Nh': [], 'Ni': [], 'Nl': []}
    entity_pattern = ""

    for entity_type in entity_types:
        for i in range(sentence_len):
            if (ne_tags[i] == 'B-' + entity_type) or (ne_tags[i] == 'B-' + entity_type):
                entity_pattern += words[i]
            elif (ne_tags[i] == 'E-' + entity_type) or (ne_tags[i] == 'S-' + entity_type):
                entity_pattern += words[i]
                ret_entity[entity_type].append(entity_pattern)
                entity_pattern = ""
    ret_cnt = 0
    for k in ret_entity:
        ret_cnt += len(list(ret_entity[k]))
    return ret_cnt


def process_tokenize(root, language):
    """
    tokenizer
    :param root: root path of dataset
    :param language: english or chinese
    :return:
    """
    for topic in sorted(os.listdir(root)):
        print('TOPIC:', topic)
        if os.path.exists(root / topic / 'articles.jsonl.gz'):
            articles = list(utils.read_jsonl_gz(root / topic / 'articles.jsonl.gz'))
        elif os.path.exists(root / topic / 'articles.jsonl'):
            articles = list(utils.read_jsonl(root / topic / 'articles.jsonl'))
        else:
            continue

        jsonl_out_path = root / topic / 'articles.tokenized.jsonl'

        out_batch = []
        for i, a in enumerate(articles):
            if language == 'chinese':
                tokenized_doc, tokenized_sent, tokenized_title, sentences = chinese_tokenizer(a['text'], a['title'])
                name_entity_num = list(map(chinese_name_entity_counter, tokenized_sent))
                title_entity_num = chinese_name_entity_counter(tokenized_title)
            elif language == 'english':
                tokenized_doc, tokenized_sent, tokenized_title, sentences = english_tokenizer(a['text'], a['title'])
                name_entity_num = list(map(english_name_entity_counter, tokenized_sent))
                title_entity_num = chinese_name_entity_counter(tokenized_title)

            else:
                tokenized_doc, tokenized_sent, tokenized_title, sentences = english_tokenizer(a['text'], a['title'])
                name_entity_num = list(map(english_name_entity_counter, tokenized_sent))
                title_entity_num = chinese_name_entity_counter(tokenized_title)

            a['tokenized_text'] = tokenized_doc.strip()
            a['tokenized_title'] = tokenized_title.strip()
            a['title_entity_num'] = title_entity_num
            a['sentences'] = [{'raw': s, 'token': ts, 'name_entity_num': nn} for s, ts, nn in
                              zip(sentences, tokenized_sent, name_entity_num)]
            out_batch.append(a)
            if i % 100 == 0:
                utils.write_jsonl(out_batch, jsonl_out_path, override=False)
                out_batch = []
                print(i)

        utils.write_jsonl(out_batch, jsonl_out_path, override=False)

        gz_out_path = root / topic / 'articles.tokenized.jsonl.gz'
        utils.gzip_file(jsonl_out_path, gz_out_path, delete_old=False)


def get_chinese_date(publish_time, time_str):
    """

    :param publish_time: publish time str
    :param time_str: search time in a sentence
    :return: time str (yyyy-mm-dd) or None
    """
    year_index = time_str.find('年') - 1
    month_index = time_str.find('月') - 1
    day_index = time_str.find('日') - 1

    def get_num(index):
        ret = ""
        while index >= 0:
            if '0' <= time_str[index] <= '9':
                ret = time_str[index] + ret
            else:
                break
            index -= 1
        return ret

    day = get_num(day_index)
    if day == "":
        return None

    year = get_num(year_index) if len(get_num(year_index)) == 4 else publish_time[:4]
    month = get_num(month_index) if get_num(month_index) != "" else publish_time[5:7]
    day = get_num(day_index) if get_num(day_index) != "" else publish_time[8:]
    return "{}-{}-{}".format(year,
                             month.zfill(2),
                             day.zfill(2))


def get_sentence_time(sentence, language, pub_time):
    """

    :param sentence: sentence to process
    :param language: chinese or english
    :param pub_time: publish time str
    :return: a list of time str
    """
    ret = []
    culture = Culture.Chinese if language == 'chinese' else Culture.English

    if language == 'chinese':
        output = get_chinese_date(pub_time, sentence)
        if output:
            ret.append(output)

    try:
        outputs = recognize_datetime(sentence, culture, reference=utils.str_to_date(pub_time))
    except:
        print('get_time format time error')
        return []

    for item in outputs:
        if item.resolution and item.resolution['values'][0]['type'] == 'date':
            ret.append(item.resolution['values'][0]['value'][:10])

    cleaned_ret = []
    for item in ret:
        try:
            y, m, d = item.split('-')
            _ = datetime.datetime(int(y), int(m), int(d))
            cleaned_ret.append(item)
        except:
            continue
    return list(set(cleaned_ret))


def process_time(root, language):
    """

    :param root: root path of dataset
    :param language: chinese or english
    :return:
    """
    for topic in os.listdir(root):
        print('TOPIC:', topic)
        if os.path.exists(root / topic / 'articles.tokenized.jsonl.gz'):
            articles = list(utils.read_jsonl_gz(root / topic / 'articles.tokenized.jsonl.gz'))
        elif os.path.exists(root / topic / 'articles.tokenized.jsonl'):
            articles = list(utils.read_jsonl(root / topic / 'articles.tokenized.jsonl'))
        else:
            continue

        jsonl_out_path = root / topic / 'articles.preprocess.jsonl'

        out_batch = []
        for i, a in enumerate(articles):
            sentences = [s['raw'] for s in a['sentences']]
            time_list = []
            pub_time = a['time'][:10]
            for sent in sentences:
                tmp_time = get_sentence_time(sent, language, pub_time)
                time_list.append(tmp_time)
            a['sentences'] = [
                {'text': s['raw'], 'token': s['token'], 'name_entity_num': s['name_entity_num'], 'time': t,
                 'pub_time': pub_time, } for s, t in
                zip(a['sentences'], time_list)]
            out_batch.append(a)

            if i % 100 == 0:
                utils.write_jsonl(out_batch, jsonl_out_path, override=False)
                out_batch = []
                print(i)

        utils.write_jsonl(out_batch, jsonl_out_path, override=False)
        gz_out_path = root / topic / 'articles.preprocess.jsonl.gz'
        utils.gzip_file(jsonl_out_path, gz_out_path, delete_old=False)


def solve(args):
    dataset_dir = pathlib.Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError('dataset not found')
    process_tokenize(dataset_dir, args.language)
    process_time(dataset_dir, args.language)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory')
    parser.add_argument('--language', required=True, help='english or chinese')
    solve(parser.parse_args())
    # python preprocess.py --language chinese --dataset datasets/chinese-example/
    # python preprocess.py --language chinese --dataset datasets/TLCN/

