{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pickle\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.utils import shuffle\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_true = pd.read_csv(\"True.csv\")\n",
    "data_false = pd.read_csv(\"Fake.csv\")\n",
    "data_true[\"label\"]= 'REAL'\n",
    "data_false[\"label\"]= 'FAKE'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.concat([data_true, data_false])\n",
    "data = shuffle(data)\n",
    "data = data.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(44898, 5)"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title      False\n",
       "text       False\n",
       "subject    False\n",
       "date       False\n",
       "label      False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = data['text']\n",
    "y = data['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)\n",
    "tfidf_train=tfidf_vectorizer.fit_transform(x_train)\n",
    "tfidf_test=tfidf_vectorizer.transform(x_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 99.44%\n"
     ]
    }
   ],
   "source": [
    "pac=PassiveAggressiveClassifier(max_iter=50)\n",
    "pac.fit(tfidf_train,y_train)\n",
    "y_pred=pac.predict(tfidf_test)\n",
    "score=accuracy_score(y_test,y_pred)\n",
    "print(f'Accuracy: {round(score*100,2)}%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('nbmodel', MultinomialNB())])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "         steps=[('tfidf',\n",
       "                 TfidfVectorizer(analyzer='word', binary=False,\n",
       "                                 decode_error='strict',\n",
       "                                 dtype=<class 'numpy.float64'>,\n",
       "                                 encoding='utf-8', input='content',\n",
       "                                 lowercase=True, max_df=1.0, max_features=None,\n",
       "                                 min_df=1, ngram_range=(1, 1), norm='l2',\n",
       "                                 preprocessor=None, smooth_idf=True,\n",
       "                                 stop_words='english', strip_accents=None,\n",
       "                                 sublinear_tf=False,\n",
       "                                 token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b',\n",
       "                                 tokenizer=None, use_idf=True,\n",
       "                                 vocabulary=None)),\n",
       "                ('nbmodel',\n",
       "                 MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],\n",
       "         verbose=False)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pipeline.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy 0.9370452858203415\n"
     ]
    }
   ],
   "source": [
    "score=pipeline.score(x_test,y_test)\n",
    "print('accuracy',score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = pipeline.predict(x_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "        FAKE       0.93      0.94      0.94      6993\n",
      "        REAL       0.94      0.93      0.93      6477\n",
      "\n",
      "    accuracy                           0.94     13470\n",
      "   macro avg       0.94      0.94      0.94     13470\n",
      "weighted avg       0.94      0.94      0.94     13470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, prediction))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('model.pkl', 'wb') as handle:\n",
    "    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
