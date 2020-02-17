# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:43:06 2020

@author: Rahul
"""
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv',',',error_bad_lines=False)
data[data['password'].isnull()]
#data=data.dropna()
data.dropna(inplace=True)
passwords_tuple=np.array(data)

#import random
#passwords_tuple=random.shuffle(passwords_tuple) #shuffling randomly for robustness

y=[labels[1] for labels in passwords_tuple]
x=[labels[0] for labels in passwords_tuple]

import seaborn as sns
sns.set_style('whitegrid')
sns.countplot(x='strength',data=data,palette='RdBu_r')

def word_char(input):
    chars=[]
    for i in input:
        chars.append(i)
    return chars

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_char)
x=vectorizer.fit_transform(x)
x.shape

vectorizer.vocabulary_
data.iloc[0,0]

feature_names = vectorizer.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=x[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

log_class=LogisticRegression(penalty='l2',multi_class='ovr')
log_class.fit(x_train,y_train)

print(log_class.score(x_test,y_test))


##Multinomial
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(x_train, y_train) #training
print(clf.score(x_test, y_test))


x_predict=np.array(["rajjan@123"])
x_predict=vectorizer.transform(x_predict)
y_pred=log_class.predict(x_predict)
print(y_pred)













