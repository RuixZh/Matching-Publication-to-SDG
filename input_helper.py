import tensorflow as tf
from collections import OrderedDict, defaultdict
import numpy as np
import re
import matplotlib.pyplot as plt
from os.path import join, exists
import pandas as pd
import heapq
import timeit
import string
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, \
    precision_recall_curve
from tensorflow.python.keras.preprocessing import sequence, text
from nltk.corpus import stopwords
from nltk import SnowballStemmer, word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

stop = stopwords.words('english')
stemmer = SnowballStemmer('english')
stop = [stemmer.stem(i) for i in stop]
stop.append('dot')


def decontracted(text):
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def clean_str(text):
    text = text.replace(u'\xa0', u' ')
    text = decontracted(text)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"[^A-Za-z]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return [stemmer.stem(i) for i in word_tokenize(text) if len(i) > 2 and stemmer.stem(i) not in stop]


def clean_keywords(keywords):
    keywords = keywords.replace(u'\xa0', u' ')
    keywords = re.split('\;|dot|\,|and|or|\.', keywords)
    keywords = [[stemmer.stem(i) for i in word_tokenize(x.strip()) if len(i) > 2 and stemmer.stem(i) not in stop] for x in keywords]
    # text = text.translate(str.maketrans('', '', string.punctuation))
    return keywords


def prepare_data():  # df=pd.read_excel('SDG.xlsx'), odf=pd.read_excel('SDG_Overview.xlsx')):
    df = pd.read_excel('all_SDG.xlsx')
    df['SDG'] = df['SDG'].apply(str)
    df = df.rename(columns=dict(zip(df.columns,['SDG','Title','Keywords','Keywords_plus','Abstract','TiKeyAbs'])))
    df = df.groupby(['Title']).agg({'Keywords':'first','Keywords_plus':'first','Abstract':'first','TiKeyAbs':'first','SDG': ','.join}).reset_index()
    mlb = MultiLabelBinarizer()
    mlb_result = mlb.fit_transform([str(df.loc[i,'SDG']).split(',') for i in range(len(df))])
    df_final = pd.concat([df[df.columns[:-1]],pd.DataFrame(mlb_result,columns=list(mlb.classes_))],axis=1)
    # label = df_final[5:].apply(int).tolist()

    # odf.Targets = pd.Series(odf.Targets).fillna(method='ffill')
    # odf.Goals = pd.Series(odf.Goals).fillna(method='ffill')
    # odf['goal_no'] = [[int(s) for s in re.split(r'[\s|.]', i) if s.isdigit()][0] for i in odf['goal_no']]
    # odf['target_no'] = [i.split()[0].split('.')[1] for i in odf['Targets']]
    # odf['Goals'] = odf['Goals'].apply(lambda x: re.sub(r'Goal\s\d+.\s(?=[A-Z])','',re.sub('\s{2}',' ', x)).strip())
    # odf['Targets'] = odf['Targets'].apply(lambda x: re.sub(r'\d+.\w+\s(?=[A-Z])','',re.sub('\s{2}',' ', x)).strip())
    # odf['Indicators'] = odf['Indicators'].apply(lambda x: re.sub(r'\d+.\w+.\w+\s(?=[A-Z])','',re.sub('\s{2}',' ', x)).strip())

    # odf = odf.groupby('Indicators').agg({'Goals':'first','SDG':'first','Targets':'first'}).reset_index()
    # odf = odf.groupby('Targets').agg({'Goals':'first','SDG':'first','Indicators':'. '.join}).reset_index()
    # odf = odf.groupby('Goals').agg({'SDG':'first', 'Targets':'. '.join, 'Indicators':'. '.join}).reset_index()
    # odf = odf[['SDG', 'Goals', 'Targets', 'Indicators']]
    return df_final  #, odf


def get_tficf(text):
    text = [' '.join(clean_str(x)) for x in text]
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=50000, stop_words='english')
    tficf = vectorizer.fit_transform(text)
    return tficf.toarray(), vectorizer.get_feature_names()


def label_sequence(df, label):
    df['goal_no'] = df['goal_no'].apply(str)
    df['goal_no'] = df['goal_no'].astype('category')
    not_in_list = df['goal_no'].cat.categories.difference(label)
    df['goal_no'] = df['goal_no'].cat.set_categories(np.hstack((label, not_in_list)), ordered=True)
    df = df.sort_values('goal_no')
    return df


def label_correlation(df, label):
    # construct the label positive correlation and negative correlation
    # goals, goals_word = get_tficf(df['Goals'].tolist())
    # targets, targets_word = get_tficf(df['Targets'].tolist())
    #indicators, indicators_word = get_tficf(df['Indicators'].tolist())
    # tficf = np.concatenate([goals, targets, indicators], axis=-1)
    ## Indicators level feature
    tficf, word_0 = get_tficf(df['Indicators'].tolist())
    level_0 = defaultdict(lambda: defaultdict(list))
    i = 0
    for idx, row in df.iterrows():
        level_0[row['goal_no']][row['target_no']].append(tficf[i])
        i += 1
    nT, nI = max([len(x) for x in level_0.values()]), max([max([len(i) for i in x.values()]) for x in level_0.values()])
    level_0 = sequence.pad_sequences(
        [sequence.pad_sequences(
            [i for i in x.values()],
            maxlen=nI, dtype='float32', padding='post', truncating='post') for x in level_0.values()],
                maxlen=nT, dtype='float32', padding='post', truncating='post')
    np.save("level_0.npy", level_0)
    json.dump(word_0, open('word_0.json', 'w', encoding='utf-8'), indent=4)

    ## Targets level feature
    label_l1 = df.groupby('Targets').agg({'goal_no':'first','Goals':'first','Indicators':'. '.join,'target_no':','.join}).reset_index()
    label_l1 = label_sequence(label_l1, label)
    tficf, word_1 = get_tficf((label_l1['Targets']+' '+label_l1['Indicators']).tolist())
    level_1 = defaultdict(list)
    i = 0
    for idx, row in label_l1.iterrows():
        level_1[row['goal_no']].append(tficf[i])
        i+=1
    level_1 = sequence.pad_sequences([x for x in level_1.values()], maxlen=nT, dtype='float32', padding='post', truncating='post')
    np.save("level_1.npy", level_1)
    json.dump(word_1, open('word_1.json', 'w', encoding='utf-8'), indent=4)

    ## Goals level feature
    label_l2 = label_l1.groupby('Goals').agg({'goal_no': 'first','Targets': '. '.join, 'Indicators': '. '.join}).reset_index()
    label_l2 = label_sequence(label_l2, label)
    level_2, word_2 = get_tficf((label_l2['Goals']+' '+label_l2['Targets']+' '+label_l2['Indicators']).tolist())
    np.save("level_2.npy", level_2)
    json.dump(word_2, open('word_2.json', 'w', encoding='utf-8'), indent=4)
    return level_0, level_1, level_2, word_0, word_1, word_2

def load_data(data_path, clean=False):
    if clean:
        data = pd.read_csv(data_path)
        data['Keywords'] = data['Keywords'].fillna('')
        # label = data.columns[5:].tolist()
        # label = list(map(str, label))
        # Y = np.array(data[label])
        # clean the publications
        data['Title'] = data['Title'].apply(lambda x: clean_str(x))
        data['Abstract'] = data['Abstract'].apply(lambda x: clean_str(x))
        data['Keywords'] = data['Keywords'].apply(lambda x: clean_str(x))
        # data.to_csv('cleaned_SDG.csv',index=False,header=True)
    else:
        data = pd.read_csv(data_path)
    label = data.columns[5:].tolist()
    label = list(map(str, label))
    Y = np.array(data[label])
    title = data['Title'].tolist()
    abstract = data['Abstract'].tolist()
    keywords = data['Keywords'].tolist()
    return title, abstract, keywords, Y, label

def load_label(overview_path=False, label=None):

    if overview_path:
        over = pd.read_csv(overview_path)
        over = label_sequence(over, label)
        level_0, level_1, level_2, word_0, word_1, word_2 = label_correlation(over, label)
    else:
        level_0, level_1, level_2 = np.load('level_0.npy'), np.load('level_1.npy'), np.load('level_2.npy')
        word_0, word_1, word_2 = json.load(open('word_0.json', 'r', encoding='utf-8')), json.load(open('word_1.json', 'r', encoding='utf-8')), json.load(open('word_2.json', 'r', encoding='utf-8'))

    return level_0, level_1, level_2, word_0, word_1, word_2


def evaluation(true_onehot_labels, predicted_onehot_labels, predicted_scores):
    pre = precision_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_onehot_labels), average='micro')
    rec = recall_score(y_true=np.array(true_onehot_labels),
                               y_pred=np.array(predicted_onehot_labels), average='micro')
    F = f1_score(y_true=np.array(true_onehot_labels),
                         y_pred=np.array(predicted_onehot_labels), average='micro')
    # Calculate the average AUC
    auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                             y_score=np.array(predicted_scores), average='micro')
    # Calculate the average PR
    prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                       y_score=np.array(predicted_scores), average='micro')
    return pre, rec, F, auc, prc

def grid_search(true_labels, eval_scores):
    '''
    # eval_scores = np.array(eval_scores)
    auc = roc_auc_score(y_true=np.array(true_labels),
                        y_score=np.array(eval_scores), average='micro')
    # precision, recall, thresholds = precision_recall_curve(np.array(true_labels), np.array(eval_scores))
    # Calculate the average PR
    prc = average_precision_score(y_true=np.array(true_labels),
                                  y_score=np.array(eval_scores), average='micro')
    print('roc_uc: {}, pr_auc: {}'.format(auc, prc))

    '''
    global eval_metrics
    max_f1 = -0.01
    eval_scores = np.array(eval_scores)
    for t in [0.5]:
        pred_label = (eval_scores > t).astype(int)
        pre, rec, f1, roc, pr = evaluation(true_labels, np.array(pred_label), eval_scores)
        if f1 > max_f1:
            max_f1 = f1
            max_t = t
            eval_metrics = (pre, rec, f1, roc, pr)
    print('Max metrics under threshold {}'.format(max_t))
    print('\nprecision: {0[0]}, recall: {0[1]}, f1: {0[2]}, roc_uc: {0[3]}, pr_auc: {0[4]}'.format(eval_metrics))
    return eval_metrics


def main(data_path=None, overview_path=None, gt=False):
    print("Loading data...")
    return load_data(data_path, overview_path, generate_tficf=gt)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1)/batch_size) + 1
    print("Total {} epochs".format(num_epochs))
    print("{} steps for each epoch".format(num_batches_per_epoch))
    print("==========")
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
