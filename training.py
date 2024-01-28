import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re

dataset = pd.read_csv('dataset_balanced_clean_mv.csv')

print('loaded dataset')
dataset = dataset.dropna()
print(dataset['feedback'].value_counts())
cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(dataset.verified_reviews)

print('vectorization')

corpus = []
stemmer = PorterStemmer()
for i in range(0, dataset.shape[0]):
  review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i]['verified_reviews'])
  review = review.lower().split()
  review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)

print('tokenization')

cv = CountVectorizer(max_features = 2500)

X = cv.fit_transform(corpus).toarray()
y = dataset['feedback'].values
pickle.dump(cv, open('countVectorizer.pkl', 'wb'))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

scaler = MinMaxScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)


pickle.dump(scaler, open('scaler.pkl', 'wb'))


model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)


print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))

pickle.dump(model_xgb, open('model_xgb.pkl', 'wb'))