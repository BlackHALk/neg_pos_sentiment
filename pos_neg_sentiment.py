
# coding: utf-8

# In[1]:


import os
import pandas as pd

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from keras.layers import *
from keras.models import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import TSNE
NB = MultinomialNB()
from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer()
vect = CountVectorizer()

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns; sns.set(font_scale=1.2)
from sklearn import svm


# In[2]:


# reading tsv file
train_data = pd.read_csv('C:\\Users\\User\\Desktop\\pos_neg_data\\all\\train.tsv',sep = "\t" ,index_col = False)


# In[3]:


train_data.head(20)


# In[4]:


#np.random.shuffle(train_data.values)
train_data.reindex(np.random.permutation(train_data.index))


# In[5]:


# Pre-processing


# In[69]:


train_data.head()


# In[7]:


# Making everything in lowercase

train_data['Phrase'] = train_data['Phrase'].str.lower()

# Replacing and removig shit

train_data = train_data.replace('[^A-Za-z0-9(),!?\'\`]', ' ', regex=True)
train_data = train_data.replace("\'s", "\'s", regex=True)
train_data = train_data.replace("\'ve", " \'ve", regex=True)
train_data = train_data.replace("n\'t", " n\'t", regex=True)
train_data = train_data.replace("\'re", " \'re", regex=True)
train_data = train_data.replace("\'d", " \'d", regex=True)
train_data = train_data.replace("\'ll", " \'ll", regex=True)
train_data = train_data.replace(",", " , ", regex=True)
train_data = train_data.replace("!", " ! ", regex=True)
train_data = train_data.replace("\(", " \( ", regex=True)
train_data = train_data.replace("\)", " \) ", regex=True)
train_data = train_data.replace("\?", " \? ", regex=True)
train_data = train_data.replace("\s{2,}", " ", regex=True)


# In[8]:


train_data.head(10)


# In[9]:


train_data.head(10)
train_data.describe()


# In[10]:


train_data.count()


# In[11]:


len(train_data)


# In[12]:


x = train_data.Phrase
y = train_data.Sentiment

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=1)


# In[13]:


x_train.count()


# In[14]:


x_test.count()


# In[15]:


y_train.count()


# In[16]:


y_test.count()


# In[17]:


s = train_data['Sentiment'].value_counts()
sns.barplot(x=s.values, y=s.index)
plt.title('Data Distribution')


# In[18]:


# vectorizing 

x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)


# In[19]:


# Naive base

NB.fit(x_train_dtm,y_train)
y_predict = NB.predict(x_test_dtm)
metrics.accuracy_score(y_test,y_predict)


# In[20]:


# Random Forest

rf = RandomForestClassifier(max_depth=10,max_features=10)
rf.fit(x_train_dtm,y_train)
rf_predict = rf.predict(x_test_dtm)
metrics.accuracy_score(y_test,rf_predict)


# In[ ]:





# In[60]:


# MLP nn sklearn

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=5000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(x_train_dtm, y_train)
print("Training set score: %f" % mlp.score(x_train_dtm, y_train))
print("Test set score: %f" % mlp.score(x_train_dtm, y_train))
mlp.classes_
pred = mlp.predict(x_test_dtm)
metrics.accuracy_score(y_test,pred)


# In[61]:


nb_accur = metrics.accuracy_score(y_test,y_predict)
rf_accur =  metrics.accuracy_score(y_test,rf_predict)
mlp_accur = metrics.accuracy_score(y_test,pred)


# In[62]:


# accu = pd.DataFrame([{"Naive base" : nb_accur, "Random Forest" : rf_accur, "MLP" : mlp_accur}])
accu = pd.DataFrame({"algo" : ["Naive base","Random Forest","MLP"],"accuracy" : [nb_accur,rf_accur,mlp_accur]})
accu


# In[63]:


sns.barplot(x=accu.accuracy, y=accu.algo,  hue="algo", data=accu)
plt.title('Data Distribution')


# In[64]:


sns.distplot(accu.accuracy, label= "accuracy")


# In[39]:


s


# In[65]:


# Fit the SVM model

model_ = svm.SVC(kernel='linear')
model_.fit(x_train_dtm,y_train)
svmPredict = model_.predict(x_test_dtm)
metrics.accuracy_score(y_test,svmPredict)


# In[66]:


svm_acc = metrics.accuracy_score(y_test,svmPredict)
accu = pd.DataFrame({"algo" : ["Naive base","Random Forest","MLP","SVM"],"accuracy" : [nb_accur,rf_accur,mlp_accur,svm_acc]})
accu


# In[67]:


sns.barplot(x=accu.accuracy, y=accu.algo,  hue="algo", data=accu)
plt.title('Data Distribution')


# In[68]:


sns.distplot(accu.accuracy, label= "accuracy")

