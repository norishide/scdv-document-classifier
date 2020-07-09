#!/usr/bin/env python
# coding: utf-8

# This code vectorize documents and then classify them.  
# livedoor corpus is used.

# ## logger and loader

# We prepare logger and loader to save intermmediate files.  
# It may takes long time to make SCDV.

# In[1]:


import datetime
import glob
import os
import json


# logger
def initialize_logger(dirpath='../data'):
    dirname = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    os.makedirs(os.path.join(dirpath, dirname), exist_ok=True)
    
    def logger(filename, obj):    
        filepath = os.path.join(dirpath, dirname, filename)
        _, ext = os.path.splitext(filename)
        if ext == '.npy':
            np.save(filepath, obj)
        elif ext == '.json':
            json.dump(obj, open(filepath, 'w'), indent=4, ensure_ascii=False)
        else :
            raise Exception

    return logger


# loader
def load_log(timestamp, filename, dirpath='../data'):
    filepath = os.path.join(dirpath, timestamp, filename)
    
    _, ext = os.path.splitext(filename)
    if ext == '.npy':
        return np.load(filepath)
    elif ext == '.json':
        return json.load(open(filepath, 'r'))
    else :
        raise Exception


# ## preprocessing

# Japanese is not separated. We separate into token and preprocess them. Then we get list of  tokens for each documents.

# In[33]:


import glob
import os
import re
from itertools import count, filterfalse
from functools import partial

import numpy as np
import pandas as pd


# In[34]:


input_train='../data/aozora_datasets_min/train_author_novel.csv'
df_train = pd.read_csv(input_train)
df_train


# In[35]:


pd.concat([df_train.groupby('author_name').novel_title.nunique(),
           df_train.author_name.value_counts()], axis=1, sort=False)


# In[69]:


filename_author_map = dict(zip(df_train.filename, df_train.author_name))
author_label_map = dict(zip(set(df_train.author_name), count())) 

train_files = glob.glob('../data/aozora_datasets_min/train_datasets/*.txt')
train_filenames = list(map(os.path.basename, train_files))
train_authors = list(map(filename_author_map.get, train_filenames))

y_train = np.array(list(map(author_label_map.get, train_authors)))
y_train, y_train.shape


# In[37]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from gensim.models import word2vec
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer


# In[56]:


import MeCab
import pandas as pd
import re
# from mojimoji import han_to_zen, zen_to_han
import numpy as np
from collections import Counter
from functools import partial, reduce
from itertools import chain, count, filterfalse, repeat
from operator import countOf, itemgetter, methodcaller

compose = lambda *funcs: reduce(lambda f, g: lambda *args, **kwargs: f(g(*args, **kwargs)), funcs)
flip = lambda f: lambda *args: f(*args[::-1])
pipe = flip(compose)


# str -> list
split_text_into_lines = methodcaller('splitlines')
# str -> str
convert_text_to_lowercase = methodcaller('lower')
# str -> str
replace_whitespace_with_blank_char = partial(re.sub, "\s", " ")
# str -> str
remove_whitespace = partial(re.sub, "\s", "")
# str -> str
strip_multiple_blank_chars_to_one = partial(re.sub, " +", " ")
# str -> str
strip_consecutive_chars_to_one = lambda char='ー': partial(re.sub, '{}+'.format(char), '{}'.format(char))

# str -> str
def insert_text_between_blank_chars(text, patterns):
    repls = map(" {} ".format, patterns)
    for pattern, repl in zip(patterns, repls):
        text = text.replace(pattern, repl)
    return text

# str -> str
def remove_whitespace_between_japanese_chars(text):
    """
    """
    chars = "[、。〃々〆〇ぁ-んゔゝゞァ-ンヴヵヶ・ーヽヾ一-龥]"
    pattern = "(?<={chars})\s+(?={chars})".format(chars=chars)
    repl = ""
    return re.sub(pattern, repl, text)

# str -> list
parse_tab_separated_text = re.compile("\t").split
noisy_symbols = list('!"#$%&\'()*+,-./:;<=>?@[]^_`{|}¢£¥§¨¬°±´¶×÷‐―‖"†‡‥…‰′″※℃Å←↑→↓⇒⇔∀∂∃∇∈∋−√∝∞∠∧∨∩∪∫∬∴∵∽≒≠≡≦≧≪≫⊂⊃⊆⊇⊥⌒─━│┃┌┏┐┓└┗┘┛├┝┠┣┤┥┨┫┬┯┰┳┴┷┸┻┼┿╂╋■□▲△▼▽◆◇○◎●◯★☆♀♂♪♭♯＃〈〉《》「」『』【】〒〓〔〕〜゛゜・＼／｀￣（）。、”’｀？！［］')

def maybe_lemmatize(mecab_node):
    for surface, feature in map(re.compile('\t').split, mecab_node.splitlines()[:-1]):
        if len(feature.split(',')) == 7:
            yield surface
        else:
            yield feature.split(',')[6]

def remove_noisy_symbols(raw_tokens):
    remove = dict.fromkeys(noisy_symbols, True).get
    return filterfalse(remove, raw_tokens)

preprocess = pipe(
    str,
#     han_to_zen,
#     partial(zen_to_han, kana=False),
    convert_text_to_lowercase,
    replace_whitespace_with_blank_char,
    remove_whitespace_between_japanese_chars,
    strip_consecutive_chars_to_one(),
    partial(insert_text_between_blank_chars, patterns=noisy_symbols),
    strip_multiple_blank_chars_to_one,
#     partial(re.sub, '\d', '0'),
    MeCab.Tagger().parse,
    maybe_lemmatize,
    remove_noisy_symbols,
    list
)


def tokenize_ja(path):
    with open(path) as f:
        lines = f.readlines()
        lines = [preprocess(line) for line in lines] 
        return list(chain.from_iterable(lines))


# In[17]:


data = pd.DataFrame(columns=['author', 'tokens'])
data['author'] = train_authors
data['tokens'] = [tokenize_ja(path) for path in train_files]
data.head(15)


# In[19]:


data.loc[data.isnull().any(axis=1)]


# In[22]:


# dump preprocessed Japanese corpus

corpus_tokenized_ja = data['tokens'].tolist()
logger = initialize_logger()
logger('corpus_tokenized_ja.json', corpus_tokenized_ja)


# In[4]:


# overview Japanese corpus on histgram
import matplotlib.pyplot as plt

corpus_tokenized_ja = load_log('20200411T092421Z','corpus_tokenized_ja.json')
len_tokens_ja = list(map(len, corpus_tokenized_ja))
plt.hist(len_tokens_ja, bins=40)
plt.yscale('log')


# In[5]:


#check the number of the corpus
len(corpus_tokenized_ja)


# ## vectorize

# For document vectorization, we use [SCDV](https://arxiv.org/pdf/1612.06778.pdf).  
# Refer to the blogs \[[1](https://qiita.com/fufufukakaka/items/a7316273908a7c400868), [2](https://qiita.com/nishiba/items/40449df6c931cca38abe)\] for Japanese explanatinon.

# In[6]:


# get word vectors by gensim
def build_word_embedding_vectors(corpus_tokenized, word2vec_parameters):
    model = word2vec.Word2Vec(corpus_tokenized, **word2vec_parameters)
    return model, model.wv.vectors, model.wv.index2word


# In[27]:


# culster word vectors by GMM
def culster_embedding_vectors(word_embeddings, gmm_parameters) -> np.ndarray:
    X = word_embeddings
    gm = GaussianMixture(**gmm_parameters)
    gm.fit(X)
    return gm.predict_proba(X)


# In[8]:


# TF-IDF takes 20 min
# Use for Japanese
from collections import Counter
from itertools import chain, repeat
from operator import countOf


def build_tfidf_selfmade(corpus, vocab):
    countup = lambda doc: list(map(countOf, repeat(doc), vocab))
    tf = np.array(list(map(countup, corpus)))
    idf = np.log(len(corpus) / (tf > 0).sum(axis=0)) + 1
    return tf, idf, tf*idf


# In[9]:


# TF-IDF
# cannot be used for JApanese
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_sklearn(corpus_tokenized, vocab):
    tfv = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', strip_accents='unicode', dtype=np.float64)
    tfv.fit(list(map(' '.join, corpus_tokenized)))
    feature_names = tfv.get_feature_names()
    idf = tfv._tfidf.idf_
    print(len(feature_names), len(idf), len(vocab))
    return np.fromiter(map(dict(zip(feature_names, idf)).get, vocab), dtype='f8')    


# In[10]:


def build_word_topic_vector(word_embedding_vectors, word_cluster_probabilities, idf_vector):
    wcv = np.einsum('ij,ik->ijk', word_embedding_vectors, word_cluster_probabilities).reshape(word_embedding_vectors.shape[0], -1)
    return np.einsum('i,ij->ij', idf_vector, wcv)


# In[15]:


# document vectors
def count(i=0):
    while True:
        yield i
        i+=1

def normalize_document_vectors(dv: np.ndarray) -> np.ndarray:
    dv_sum = np.square(dv).sum(axis=1)
    dv_l2norm = np.sqrt(dv_sum)
    return np.einsum('ij,i->ij', dv, 1.0/dv_l2norm)

def build_document_vectors(vocabulary, corpus_tokenized, word_topic_vectors):
    vocabulary_idx_map = dict(zip(vocabulary, count()))
    
    document_vecotrs = []
    for doc in corpus_tokenized:
        doc_idx = list(filter(lambda idx: idx is not None, map(vocabulary_idx_map.get, doc)))
        document_vecotrs.append(np.einsum('ij->j',word_topic_vectors[doc_idx]))
    
    return normalize_document_vectors(np.array(document_vecotrs))


# In[16]:


# SCDV
def make_sparse(document_vectors, param):
    t = 0.5 * (np.abs(np.min(document_vectors, axis=1).mean()) + np.abs(np.max(document_vectors, axis=1).mean()))
    sparsity_threshold = param * t
    mask = np.abs(document_vectors) < sparsity_threshold
    return np.where(mask, 0.0, document_vectors)


# In[18]:


corpus_tokenized = corpus_tokenized_ja


# In[42]:


word2vec_parameters = {
    'size':200,   # Word vector dimensionality
    'min_count':20,    # Minimum word count
    'workers':40,    # Number of threads to run in parallel
    'window':10,    # Context window size
    'sample':1e-3,    # Downsample setting for frequent words
    'hs':0,
    'sg':1,
    'negative':10,
    'iter':25,
    'seed':1
}

word_embedding_model, word_embedding_vectors, vocabulary = build_word_embedding_vectors(corpus_tokenized, word2vec_parameters)


# In[45]:


logger = initialize_logger()
logger('word_embedding_vectors.npy', word_embedding_vectors)
logger('vocabulary.json', vocabulary)
word_embedding_model.save('word_embedding_model.model')


# In[17]:


timestamp = '20200411T140408Z'
word_embedding_vectors = load_log(timestamp,'word_embedding_vectors.npy')
vocabulary = load_log(timestamp,'vocabulary.json')
word_embedding_vectors.shape, len(vocabulary)


# In[47]:


gmm_parameters = {
    'n_components':60,
    'random_state':42,
    'covariance_type':'tied',
    'init_params':'kmeans',
    'max_iter':50
}

word_cluster_probability_matrix = culster_embedding_vectors(word_embedding_vectors, gmm_parameters)


# In[48]:


logger('word_cluster_probabilities.npy', word_cluster_probability_matrix)


# In[20]:


timestamp = '20200411T140408Z'
word_cluster_probability_matrix = load_log(timestamp,'word_cluster_probabilities.npy')
word_cluster_probability_matrix.shape


# In[21]:


tf, idf, tfidf = build_tfidf_selfmade(corpus_tokenized, vocabulary)


# In[22]:


logger = initialize_logger()
logger('tf_matrix.npy', tf)
logger('idf_vector.npy', idf)
logger('tfidf_matrix.npy', tfidf)


# In[23]:


timestamp = '20200412T080115Z'
tf_matrix = load_log(timestamp,'tf_matrix.npy')
idf_vector = load_log(timestamp,'idf_vector.npy')
tfidf_matrix = load_log(timestamp,'tfidf_matrix.npy')
tf_matrix.shape, idf_vector.shape, tfidf_matrix.shape


# In[ ]:


word_topic_vectors = build_word_topic_vector(word_embedding_vectors, word_cluster_probability_matrix, idf_vector)


# In[24]:


document_vectors = build_document_vectors(vocabulary, corpus_tokenized_ja, word_topic_vectors)
param = 0.04
scdv = make_sparse(document_vectors, param)


# In[25]:


# logger = initialize_logger()
logger('scdv.npy', scdv)


# In[26]:


timestamp = '20200412T080115Z'
scdv = load_log(timestamp,'scdv.npy')
scdv.shape


# In[ ]:





# ## Validate with SCDV

# In[70]:


X = scdv
y = y_train
X.shape, y.shape


# In[71]:


def kfold_splitter(n_samples, n_folds, rng):
    shuffled_fold_indices = rng.permutation(np.arange(n_samples) % n_folds)
    for k in range(n_folds):
        mask = shuffled_fold_indices == k
        yield tuple(map(np.flatnonzero, (mask, ~mask)))


# In[72]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

clf = RandomForestClassifier(random_state=0, max_depth=None, max_features='sqrt',
                             class_weight='balanced', n_jobs=-1)


# In[73]:


n_fold = 5
n_samples = y.size
rng = np.random.RandomState(42)

accuracy_train = []
accuracy_test = []

for test_idx, train_idx in kfold_splitter(n_samples, n_fold, rng):
    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_train)
    bas = balanced_accuracy_score(y_train, y_pred)
    accuracy_train.append(bas)
    
    y_pred = clf.predict(X_test)
    bas = balanced_accuracy_score(y_test, y_pred)
    accuracy_test.append(bas)

accuracy_train = np.array(accuracy_train)
accuracy_test = np.array(accuracy_test)


# In[74]:


accuracy_train


# In[75]:


accuracy_test


# In[76]:


np.average(accuracy_test)


# In[ ]:





# ## Validate with TF-IDF

# In[46]:


X = tfidf_matrix
X.shape, y.shape


# In[47]:


def kfold_splitter(n_samples, n_folds, rng):
    shuffled_fold_indices = rng.permutation(np.arange(n_samples) % n_folds)
    for k in range(n_folds):
        mask = shuffled_fold_indices == k
        yield tuple(map(np.flatnonzero, (mask, ~mask)))


# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

clf = RandomForestClassifier(random_state=0, max_depth=None, max_features='sqrt',
                             class_weight='balanced', n_jobs=-1)


# In[49]:


n_fold = 5
n_samples = y.size
rng = np.random.RandomState(42)

accuracy_train = []
accuracy_test = []

for test_idx, train_idx in kfold_splitter(n_samples, n_fold, rng):
    y_test = y[test_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    X_train = X[train_idx]
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_train)
    bas = balanced_accuracy_score(y_train, y_pred)
    accuracy_train.append(bas)
    
    y_pred = clf.predict(X_test)
    bas = balanced_accuracy_score(y_test, y_pred)
    accuracy_test.append(bas)

accuracy_train = np.array(accuracy_train)
accuracy_test = np.array(accuracy_test)


# In[50]:


accuracy_train


# In[51]:


accuracy_test


# In[52]:


np.average(accuracy_test)


# In[ ]:





# ## predict test data

# ### preprocess

# In[54]:


import glob, os


test_files = glob.glob('../data/aozora_datasets_min/test_datasets/*.txt')
len(test_files)


# In[60]:


data_test = pd.DataFrame(columns=['filename','tokens'])
data_test['filename'] = list(map(os.path.basename, test_files))
data_test['tokens'] = [tokenize_ja(path) for path in test_files]
data_test.head(15)


# In[62]:


data_test.loc[data_test.isnull().any(axis=1)]


# In[63]:


# dump preprocessed data
corpus_tokenized_ja_test = data_test['tokens'].tolist()
logger('corpus_tokenized_ja_test.json', corpus_tokenized_ja_test)


# In[64]:


# overview japanese corpus
import matplotlib.pyplot as plt

timestamp = '20200412T080115Z'
corpus_tokenized_ja_test = load_log(timestamp,'corpus_tokenized_ja_test.json')
len_tokens_ja = list(map(len, corpus_tokenized_ja_test))
plt.hist(len_tokens_ja, bins=40)
plt.yscale('log')


# ### vectorize test corpus

# In[65]:


# scdv ignoring unknown tokens
document_vectors = build_document_vectors(vocabulary, corpus_tokenized_ja_test, word_topic_vectors)
param = 0.04
scdv_ignr_unk = make_sparse(document_vectors, param)
scdv_ignr_unk.shape


# ### predict test data

# In[77]:


X_test = scdv_ignr_unk
X_test.shape


# In[78]:


y_pred_proba = clf.predict_proba(X_test)


# In[82]:


y_pred_proba.shape


# In[79]:


plt.hist(list(map(max, y_pred_proba)), bins=20)
# plt.hist(y_pred_proba.max(axis=1), bins=20)


# In[80]:


y_pred = y_pred_proba.argmax(axis=1)
u, counts = np.unique(y_pred, return_counts=True)
plt.bar(u, counts)
plt.xticks(u);


# ### set authors

# In[95]:


label_author_map = dict([(v,k) for k,v in author_label_map.items()])
author_pred = list(map(label_author_map.get, y_pred))


# In[89]:


data_test.assign(predicted_author = author_pred)


# In[90]:


filename_author_map_test = dict(zip(data_test['filename'], data_test['predicted_author']))


# In[107]:


import pandas as pd

input_test='../data/aozora_datasets_min/test_author_novel.csv'
df_test = pd.read_csv(input_test)
df_test['author'] = list(map(filename_author_map_test.get, df_test['filename']))
df_test['author_id'] = list(map(            dict(zip(df_train['author_name'], df_train['author_id'])).get, df_test['author']            ))
df_test


# In[126]:


df_test.to_csv('../data/test_author_novel_predicted.csv', index=False)


# In[ ]:





# ### check predicted results

# In[109]:


df_test.loc[200,:]


# In[110]:


df_test.index


# In[111]:


rng = np.random.RandomState(0)


# In[112]:


df_test.loc[rng.choice(df_test.index,10),:]


# In[124]:


path_test_samples = df_test.filename.sample(1, replace=False)
path_test_samples


# In[125]:


list(map(filename_author_map_test.get, path_test_samples))


# In[120]:


path = '56314.txt'
path_test = '../data/aozora_datasets_min/test_datasets/{}'.format(path)
with open(path_test) as f:
    print(f.read())


# In[121]:


filenametrain='56315.txt'
path_test = '../data/aozora_datasets_min/train_datasets/{}'.format(filenametrain)
with open(path_test) as f:
    print(f.read())


# In[123]:


df_train[df_train.filename == '56315.txt']


# In[ ]:




