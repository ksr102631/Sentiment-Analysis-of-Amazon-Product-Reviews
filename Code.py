# PM OPIM -5604 FINAL PROJECT
# @ author : SREERAM KASARLA

#### REFERENCES 

#https://dev.to/rodolfoferro/sentiment-analysis-on-trumpss-tweets-using-python-?utm_campaign=Data%2BElixir&utm_medium=email&utm_source=Data_Elixir_149
#http://t-redactyl.io/blog/2017/04/applying-sentiment-analysis-with-vader-and-the-twitter-api.html
#https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
#https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/util.py




#Tagging sentiment with Text Blob

import pandas as pd
train = pd.read_csv("7817_1.csv")
import re
from nltk.corpus import stopwords

import string
#string.punctuation

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a rawreview), and 
    # the output is a single string (a preprocessedreview)
    #
    # 1. Remove HTML
    review_text = raw_review
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words1 = [w for w in words if not w in stops]   
    
    # 5b. Remove punctuation
    meaningful_words2 = [w for w in meaningful_words1 if w not in string.punctuation]
    
    # 5c. Lemmitization is more accurate 
    meaningful_words3 = [lem.lemmatize(w) for w in meaningful_words2]
    
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words3))   

def analyze_sentiment(text):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(review_to_words(text))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1	
	
train['Sentiment_TextBlob'] = np.array([ analyze_sentiment(text) for text in train['reviews.text'] ])


train.to_csv("TB2_Cleaned.csv",columns =['reviews.text','Sentiment_TextBlob'], index = False)



# Tagging sentiment with Vader 

df = pd.read_csv('7817_1.csv')


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


score_compound = []

text = []

for i in range(0, len(df)):
    test.append(df['reviews.text'][i])
    score_compound.append(analyzer.polarity_scores(df['reviews.text'][i])['compound'])
	
	
from pandas import Series,DataFrame

new_dataframe = DataFrame({'Text': text,'Compound': score_compound})

new_dataframe.to_csv("Vaderanalysis.csv",index = False)


## Building ML Models using Text as Features



import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


train = pd.read_csv("train.csv")


## Accuracy metric - Multiclass Logloss function -source cited in references 

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
	


label_enc = preprocessing.LabelEncoder()
y = label_enc.fit_transform(train.Sentiment.values)


np.unique(y)

# 70-30 , Training and validation split


xtrain, xvalid, ytrain, yvalid = train_test_split(train.reviews.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.3, shuffle=True)
											


#print (xtrain.shape)
#print (xvalid.shape)

# Using TF-IDF as features

tfv = TfidfVectorizer(min_df=1,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')



## Fit and transform reviews text to sparse TF-IDF features matrix
			
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)


#Using Bag of Words as features

ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv =  ctv.transform(xtrain) 
xvalid_ctv = ctv.transform(xvalid)


## Generally BOW is naive so we prefer to go with TF-IDF features to build models 


# ML1 - Logistic Regression

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# 0.362

# ML2  - Naive Bayes

clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# 0.385
## Logistic Regression works better when there are few features and the data is scaled 

## Using SVD to keep only 120 (usually taken) components(Sort of Dimensionality reduction technique)


svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. 
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)


## Fitting a simple SVM
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#0.353



## Testing the CV scores of Gradient boosting and tree based models

## Random Forest with 10 fold CV

from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier()
kfold = KFold(n_splits=10, random_state= 123)
result = cross_val_score(rf1, xtrain_svd_scl, ytrain, cv=kfold, scoring='neg_log_loss')
print(np.median(result))
# -0.95



## Decision Tree with 10 fold CV

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
kfold = KFold(n_splits=10, random_state= 123)
result = cross_val_score(rf1, xtrain_svd_scl, ytrain, cv=kfold, scoring='neg_log_loss')
print(np.median(result))
# -4.495



## Fitting Logistic Regression model with features treated with SVD and scaled
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)
print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# 0.290


# Parameter Tuning for Logistic Regression

clf = LogisticRegression(C= 0.1,random_state=123)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.329



clf = LogisticRegression(C= 0.001,random_state=123)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.894


clf = LogisticRegression(C= 10,random_state=123)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.342 




clf = LogisticRegression(C= 6,random_state=123)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

# logloss: 0.321


clf = LogisticRegression(C= 3,random_state=123)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.302





clf = LogisticRegression(C= 2)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
# logloss: 0.296

clf = LogisticRegression(C= 1)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.294


# C in Logistic Regression is the inverse regularization pararmeter, less C, more is the regularization 

## Optimal value of C is 1.0 for which the logloss is the least




### Learning rate

#200
xtrain_svd_scl_200 =  xtrain_svd_scl[:200]

ytrain_200 = ytrain[:200]

clf = LogisticRegression(C= 1,random_state=123)
clf.fit(xtrain_svd_scl_200,ytrain_200)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
#logloss: 0.451 


#400

xtrain_svd_scl_400 =  xtrain_svd_scl[:400]

ytrain_400 = ytrain[:400]

clf = LogisticRegression(C= 1,random_state=123)
clf.fit(xtrain_svd_scl_400,ytrain_400)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
# logloss: 0.364 


#600

xtrain_svd_scl_600 =  xtrain_svd_scl[:600]

ytrain_600 = ytrain[:600]

clf = LogisticRegression(C= 1,random_state=123)
clf.fit(xtrain_svd_scl_600,ytrain_600)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.333 


#800

xtrain_svd_scl_800 =  xtrain_svd_scl[:800]

ytrain_800 = ytrain[:800]

clf = LogisticRegression(C= 1,random_state=123)
clf.fit(xtrain_svd_scl_800,ytrain_800)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.303 

#1000

xtrain_svd_scl_1000 =  xtrain_svd_scl[:1000]

ytrain_1000 = ytrain[:1000]

clf = LogisticRegression(C= 1,random_state=123)
clf.fit(xtrain_svd_scl_1000,ytrain_1000)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions

#logloss: 0.296 

#1117


clf = LogisticRegression(C= 1,random_state=123)
clf.fit(xtrain_svd_scl,ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#logloss: 0.294 



#### 10 fold CV

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 

## CV
logreg = LogisticRegression()  
#logreg.fit(X,y)
kfold = KFold(n_splits=10, random_state= 123)
result = cross_val_score(logreg, xtrain_svd_scl, ytrain, cv=kfold, scoring='neg_log_loss')
print(np.median(result))

#-0.306929636089














