from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import input_helpers as iph
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import itertools

title, abstract, keywords, Y = iph.main('SDG.csv')

def train_doc2vec():
    kws = [list(itertools.chain.from_iterable(x)) for x in keywords]
    paper = [title[i]+abstract[i]+kws[i] for i in range(len(kws))]
    paper = [' '.join(x) for x in paper]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(paper)]
    model_dbow = Doc2Vec(documents, dm=0, vector_size=300, negative=5, hs=0, min_count=2, worker=4,sample=0, alpha=0.025, min_alpha=0.001)
    model_dbow.save("doc2vec.model")
    return model_dbow, Y

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def doc2vec_classification():
    model_dbow = Doc2Vec.load("doc2vec.model")
    X = [model_dbow.docvecs[i] for i in range(len(Y))]
    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.4, random_state=0)

    pred = []
    for i in range(Y.shape[1]):
        lr = LogisticRegression(C=1.0, penalty='l2')
        lr.fit(train_X, train_y.T[i])
        y_p = lr.predict(test_X)
        pred.append(y_p)

    micro_pre = precision_score(y_true=test_y, y_pred=np.array(pred).T, average='micro')  # 0.484375
    micro_rec = recall_score(y_true=test_y, y_pred=np.array(pred).T, average='micro')  # 0.0080
    micro_F = f1_score(y_true=test_y, y_pred=np.array(pred).T, average='micro')  # 0.01574
    micro_auc = roc_auc_score(y_true=test_y, y_score=np.array(pred).T, average='micro')  # 0.50325
    micro_prc = average_precision_score(y_true=test_y, y_score=np.array(pred).T, average='micro')  # 0.15244
    print('Max_metrics:\nprecision: {0[0]}, recall: {0[1]}, f1: {0[2]}, roc_uc: {0[3]}, pr_auc: {0[4]}'.format([micro_pre, micro_rec, micro_F, micro_auc,micro_prc]))


def tfidf_classification():
    import itertools
    keywords = [list(itertools.chain.from_iterable(x)) for x in keywords]
    text = [title[i] + abstract[i] + keywords[i] for i in range(len(title))]
    vectorizer = TfidfVectorizer(min_df=2)
    tfidf = vectorizer.fit_transform([' '.join(x) for x in text])
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    train_X, test_X, train_y, test_y = train_test_split(tfidf, Y, test_size=0.2, random_state=0)
    pred = []
    scores = []
    for i in range(Y.shape[1]):
        lr = LogisticRegression(C=1.0, penalty='l2')
        lr.fit(train_X, train_y.T[i])
        y_p = lr.predict(test_X)
        scores.append(lr.predict_proba(test_X))
        pred.append(y_p)

    micro_pre = precision_score(y_true=test_y, y_pred=np.array(pred).T, average='micro')  # 0.8234
    micro_rec = recall_score(y_true=test_y, y_pred=np.array(pred).T, average='micro')  # 0.6567
    micro_F = f1_score(y_true=test_y, y_pred=np.array(pred).T, average='micro')  # 0.7667
    micro_auc = roc_auc_score(y_true=test_y, y_score=np.array(pred).T, average='micro')  # 0.8234
    micro_prc = average_precision_score(y_true=test_y, y_score=np.array(pred).T, average='micro')  #0.6566


# train_doc2vec()
doc2vec_classification()
# tfidf_classification()
