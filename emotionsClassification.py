import pandas as pd
import pickle


df = pd.read_csv("datasets/dataset.txt",header=None, sep=";", names=["Comment","Emotion"], encoding="utf-8")
print(df.shape)
print(df.head())
print(df['Emotion'].value_counts())

dfJoy = df[df['Emotion']=='joy'].sample(719)
dfSadness = df[df['Emotion']=='sadness'].sample(719)
dfAnger = df[df['Emotion']=='anger'].sample(719)
dfFear = df[df['Emotion']=='fear'].sample(719)
dfLove = df[df['Emotion']=='love'].sample(719)
dfSurprise = df[df['Emotion']=='surprise'].sample(719)

dfEqual=pd.concat([dfJoy,dfSadness,dfAnger,dfFear,dfLove,dfSurprise],axis=0)
print(dfEqual.head())

print(dfEqual['Emotion'].value_counts())

dfEqual['label_num'] = dfEqual['Emotion'].map({
    'joy' : 0,
    'sadness': 1,
    'anger': 2,
    'fear': 3,
    'love':4,
    'surprise':5
})

print(dfEqual.head())

print(dfEqual.info())


dfEqual.Comment = dfEqual.Comment.fillna('')


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dfEqual.Comment,
    dfEqual.label_num,
    test_size=0.2, # 20% samples will go to test dataset
    random_state=2022,
    stratify=dfEqual.label_num
)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)

print(X_train.head())
print(y_train.value_counts())

print(X_train.head())
print(y_train.value_counts())
print(y_test.value_counts())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

#1. create a pipeline object
clf = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),
     ('KNN', KNeighborsClassifier())
])

#2. fit with X_train and y_train
clf.fit(X_train, y_train)

#3. get the predictions for X_test and store it in y_pred
y_pred = clf.predict(X_test)

#4. print the classfication report
print(classification_report(y_test, y_pred))

from sklearn.naive_bayes import MultinomialNB

#1. create a pipeline object
clf = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),
     ('Multi NB', MultinomialNB())
])

#2. fit with X_train and y_train
clf.fit(X_train, y_train)

#3. get the predictions for X_test and store it in y_pred
y_pred = clf.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


print(y_pred)

from sklearn.ensemble import RandomForestClassifier

#1. create a pipeline object
clf = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()),        #using the ngram_range parameter
     ('Random Forest', RandomForestClassifier())
])

#2. fit with X_train and y_train
clf.fit(X_train, y_train)

#3. get the predictions for X_test and store it in y_pred
y_pred = clf.predict(X_test)


#4. print the classfication report
print(classification_report(y_test, y_pred))


### model_save ###
model = "emotions.pkl"

pickle.dump(clf,open(model,'wb'))