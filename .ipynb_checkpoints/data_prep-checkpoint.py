#!/usr/bin/env python
# coding: utf-8

# In[91]:


import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import string 
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer as stemmer 
import nltk
from nltk.corpus import stopwords
from collections import Counter
import nltk.stem

# Special vocabulary module from shoyu
import vocabulary_hdp as vocab


# In[69]:


stemmer = stemmer("english")


# In[70]:


def preprocess(doc):
    return [stemmer.stem(WordNetLemmatizer().lemmatize(w, pos='v')) for w in doc.translate(str.maketrans('','', string.punctuation)).lower().split(' ')]

def rm_stopwords_and_short_words(words):
    results = []
    for i in words:
        if not i in stopwords1 and len(i)  > 3:
            results.append(i)
    return results

def full_preprocess(doc):
    return rm_stopwords_and_short_words(preprocess(doc))


# In[71]:


nltk.download('wordnet')
nltk.download('stopwords')


# In[72]:


df = pd.read_csv("tm_test_data.csv")


# In[73]:


stopwords1 = stopwords.words('english')


# In[74]:


tokenized_df = [full_preprocess(i) for i in df.abstract]


# In[75]:


all_words = [i for sublist in tokenized_df for i in sublist]
all_words =  list(set(all_words))


# In[76]:


# Filter out tokens that appear in fewer than 3 abstracts and tokens that appear in more than half the abstracts 
all_words_counts = np.zeros(len(all_words))

for k,i in enumerate(all_words):
    for j in tokenized_df:
        if i in j:
            all_words_counts[k] += 1 
            
word_counts_dict = list(zip(all_words, list(all_words_counts)))
word_counts_dict_ab = list(filter(lambda x: x[1] > 3, word_counts_dict))
word_counts_dict_ab2 = list(filter(lambda x: x[1] < len(tokenized_df)/2, word_counts_dict_ab))
final_dict = [i[0] for i in word_counts_dict_ab2]


# In[77]:


tokenized_df_ab = []
for i in tokenized_df:
    tokenized_df_ab.append([j for j in i if j in final_dict])
    


# In[92]:


voca = vocab.Vocabulary()
docs = [voca.doc_to_ids(doc) for doc in tokenized_df_ab]


# In[93]:


# Final result - list (docs) where each element is set of words represented by numbers in a global vocabulary (final dict)
docs


# In[ ]:




