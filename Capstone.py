
# coding: utf-8

# In[6]:





# In[7]:





# In[2]:


# Step 0: Receiving and reading the file.
import pandas as pd
xyz = pd.read_csv('../yelp/yelp-dataset/yelp_review.csv', nrows=200000)


# In[3]:



xyz.head()
xyz.shape



# In[4]:


# taking relevant columns from the reviews
review = xyz[['text', 'stars']]
review.head()


# In[5]:


# will help to check how many reviews are there per rating
review.stars.value_counts()


# In[6]:


# The distributin of the rating shows that the classes or output are highly imbalanced, with more number of reviews towards 
#higher ratings. The pie plot below gives the details.


# In[7]:


import matplotlib.pyplot as plt

# Pie chart
labels = ['1 star', '2 star', '3 star', '4 star', "5 star"]
sizes = [16557, 22779, 27375, 46788, 86501] 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[8]:


X = review["text"]
y = review.stars
X.shape
y.shape


# In[9]:


#STEP 1/2: PREPROCESSING AND FEATURES EXTRACTION.
#STEMMING OF DOC USING NLTK

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])


# In[10]:



# WILL USE TF-IDF VECTORIZER, WHICH IS COMBINATION OF COUNT VECTORIZER AND TF IDF TRANSFORMER
# preprocessing and feature extraction
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

#UNIGRAM
vectorizer_1 = StemmedTfidfVectorizer(stop_words='english')


# In[ ]:


X_train_dtm = vectorizer_1.fit_transform(X_train)
#dtm is data term matrix
#do fitting and transfrom in single step


# In[ ]:


tokens = vectorizer_1.get_feature_names()
print(len(tokens))
# number of features in unigram


# In[ ]:


X_test_dtm = vectorizer_1.transform(X_test)


# In[ ]:


# REPEATING WITH BIGRAM METHOD
vectorizer_2 =  StemmedTfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_train_dtm_2 = vectorizer_2.fit_transform(X_train)


# In[15]:


tokens_2 = vectorizer_2.get_feature_names()
print(len(tokens_2))
#number of features in bigram


# In[ ]:


X_test_dtm_2 = vectorizer_2.transform(X_test)


# In[17]:


print(tokens_2[20000:20059])


# In[18]:


#STEP 3: SUPERVISED LEARNING/ EVALUATION
# MULTINOMIAL NAIVE BAYES
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
nb1 = MultinomialNB()

nb1.fit(X_train_dtm, y_train)



# In[19]:


nb2 = MultinomialNB()
nb2.fit(X_train_dtm_2, y_train)


# In[ ]:


y_pred_nb1 = nb1.predict(X_test_dtm)


# In[ ]:


y_pred_nb2 = nb2.predict(X_test_dtm_2)


# In[22]:


from sklearn.metrics import f1_score
#F1 score for unigram NB
f1_score(y_test, y_pred_nb1, average= 'weighted')


# In[23]:


F1_nb = round(f1_score(y_test, y_pred_nb1, average= 'weighted'),5)
print(F1_nb)


# In[24]:


# F1 score for bigram NB
# to ignore warning due to classes with no predictions made

f1_score(y_test, y_pred_nb2, average= 'weighted')


# In[25]:


F2_nb = round(f1_score(y_test, y_pred_nb2, average= 'weighted'),5)
print(F2_nb)


# In[26]:


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(class_weight='balanced')
lr1.fit(X_train_dtm, y_train)


# In[ ]:



y_pred_lr1 = lr1.predict(X_test_dtm)


# In[28]:


lr2 = LogisticRegression(class_weight='balanced')
lr2.fit(X_train_dtm_2, y_train)


# In[ ]:


y_pred_lr2 = lr2.predict(X_test_dtm_2)


# In[30]:


# F1 score for unigram LR
f1_score(y_test, y_pred_lr1, average= 'weighted')


# In[31]:


F1_lr = round(f1_score(y_test, y_pred_lr1, average= 'weighted'),5)
print(F1_lr)


# In[32]:


# F1 score for bigram LR
f1_score(y_test, y_pred_lr2, average= 'weighted')


# In[33]:


F2_lr = round(f1_score(y_test, y_pred_lr2, average= 'weighted'),5)
print(F2_lr)


# In[ ]:


# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

param_grid = {'max_depth': [10, 20, 40, 50],
 'min_samples_leaf': [5,10,15],
 'min_samples_split': [5,10,15]}
clf = DecisionTreeClassifier(class_weight="balanced")


# In[ ]:


grid_obj1 = GridSearchCV(clf, param_grid, scoring="f1_weighted")


# In[ ]:


grid_fit1 = grid_obj1.fit(X_train_dtm, y_train)


# In[ ]:


grid_best_1 = grid_fit1.best_params_


# In[36]:


print(grid_best_1)


# In[ ]:


grid_best_obj = grid_fit1.best_estimator_
# model classifier with best estimated parameters for unigram


# In[ ]:


y_pred_dt1 = grid_best_obj.predict(X_test_dtm)


# In[39]:


# F1 score for unigram DT
f1_score(y_test, y_pred_dt1, average= 'weighted')


# In[84]:


F1_dt = round(f1_score(y_test, y_pred_dt1, average= 'weighted'),5)
print(F1_dt)


# In[ ]:


grid_obj2 = GridSearchCV(clf, param_grid, scoring="f1_weighted")


# In[ ]:


grid_fit2 = grid_obj2.fit(X_train_dtm_2, y_train)


# In[ ]:


grid_best_2 = grid_fit2.best_params_


# In[43]:



print(grid_best_2)


# In[ ]:


grid_best_obj2 = grid_fit2.best_estimator_
# model classifier with best estimated parameters for biigram


# In[ ]:


y_pred_dt2 = grid_best_obj2.predict(X_test_dtm_2)


# In[46]:


# F1 score for biigram DT
f1_score(y_test, y_pred_dt2, average= 'weighted')


# In[85]:


F2_dt = round(f1_score(y_test, y_pred_dt2, average= 'weighted'),5)
print(F2_dt)


# In[ ]:


# SUPPORT VECTOR MACHINES
from sklearn import svm
clf = svm.SVC(class_weight="balanced")
param_grid = {'C': [0.1,1,4,6,8,10,11,12], 
          'kernel': ['linear','rbf'],
         "gamma":[0.001, 0.01, 0.1, 1]}


grid_obj1 = GridSearchCV(clf, param_grid, scoring="f1_weighted")


# In[ ]:


grid_fit1 = grid_obj1.fit(X_train_dtm, y_train)


# In[49]:


grid_fit1.best_params_


# In[ ]:


grid_best_obj1 = grid_fit1.best_estimator_
# model classifier with best estimated parameters for unigram


# In[ ]:


y_pred_svm = grid_best_obj1.predict(X_test_dtm)


# In[52]:


# F1 score for unigram SVM
f1_score(y_test, y_pred_svm, average= 'weighted')


# In[86]:


F1_svm = round(f1_score(y_test, y_pred_svm, average= 'weighted'),5)
print(F1_svm)


# In[ ]:


grid_fit2 = grid_obj1.fit(X_train_dtm_2, y_train)


# In[54]:


grid_fit2.best_params_


# In[ ]:


grid_best_obj2 = grid_fit2.best_estimator_
# model classifier with best estimated parameters for biigram


# In[ ]:


y_pred_svm2 = grid_best_obj2.predict(X_test_dtm_2)


# In[57]:


# F1 score for biigram SVM
f1_score(y_test, y_pred_svm2, average= 'weighted')


# In[88]:


F2_svm = round(f1_score(y_test, y_pred_svm2, average= 'weighted'),5)
print(F2_svm)


# In[ ]:


# RANDOM FOREST CLASSIFICATION
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier
param_grid = {'max_depth': [10, 20, 40, 50, 60,None],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [20, 50, 100, 200]}
clf = RandomForestClassifier(class_weight="balanced")

grid_obj1 = GridSearchCV(clf, param_grid, scoring="f1_weighted")


# In[ ]:


grid_fit1 = grid_obj1.fit(X_train_dtm, y_train)


# In[ ]:


grid_best_1 = grid_fit1.best_params_


# In[61]:


print(grid_best_1)


# In[ ]:


grid_best_obj1 = grid_fit1.best_estimator_


# In[ ]:


y_pred_rf = grid_best_obj1.predict(X_test_dtm)


# In[64]:


# F1 score for unigram RF
f1_score(y_test, y_pred_rf, average= 'weighted')


# In[89]:


F1_rf = round(f1_score(y_test, y_pred_rf, average= 'weighted'),5)
print(F1_rf)


# In[ ]:


grid_fit2 = grid_obj1.fit(X_train_dtm_2, y_train)


# In[ ]:


grid_best_2 = grid_fit2.best_params_


# In[67]:


print(grid_best_2)


# In[ ]:


grid_best_obj2 = grid_fit2.best_estimator_


# In[ ]:


y_pred_rf2 = grid_best_obj2.predict(X_test_dtm_2)


# In[70]:


# F1 score for bigram RF
f1_score(y_test, y_pred_rf2, average= 'weighted')


# In[90]:


F2_rf = round(f1_score(y_test, y_pred_rf2, average= 'weighted'),5)
print(F2_rf)


# In[93]:


import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 5
F1_Score_unigram = (F1_nb, F1_lr, F1_dt, F1_svm, F1_rf)
F1_Score_bigram = (F2_nb, F2_lr, F2_dt, F2_svm, F2_rf)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
 
rects1 = plt.bar(index, F1_Score_unigram, bar_width,
                 alpha=opacity,
                 color='r',
                 label='F1_unigram')
 
rects2 = plt.bar(index + bar_width, F1_Score_bigram, bar_width,
                 alpha=opacity,
                 color='b',
                 label='F1_bigram')
 
plt.xlabel('Type of Classfication model')
plt.ylabel('Scores')
plt.title('')
plt.xticks(index + bar_width, ('NB', 'LR', 'DT', 'SVM', "RF"))
plt.legend()

