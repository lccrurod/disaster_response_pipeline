import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Load table messages_treated from the database filepath.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_treated', engine)
    X = df['message']
    obj_cols = list(df.columns)[4:]
    Y = df[obj_cols]
    return X, Y, obj_cols


def tokenize(text):
    """
    Remove capitalization, stops words, and return list of clean word tokens.
    """
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens

class VerbCountExtractor(BaseEstimator, TransformerMixin):
    """
    Estimator to be used in the text processing ML pipeline.
    """

    def verb_count(self, text):
        """
        Get the number of verbs in a given text.
        """
        pos_tags = nltk.pos_tag(tokenize(text))
        return sum([1 for tag in pos_tags if tag[0:2] == 'VB'])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.verb_count)
        return pd.DataFrame(X_tagged)

def model_pipeline():
    """
    Instantiate the ML pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('cvect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('vert_count', VerbCountExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])

    return pipeline

def build_model():
    pipeline = model_pipeline()
    
    parameters = {'features__text_pipeline__tfidf__use_idf':(True, False),
              'clf__n_estimators':[5, 10]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def classification_metrics(Y_true, Y_pred, category_names):
    """
    Get a dataframe with f1-score and accuracy score for classification across category_names.
    """
    metrics_dict = {'f1_score':[],
                    'accuracy':[],
                    'category':[]}
    for n, col in enumerate(category_names):
        metrics_dict['f1_score'].append(f1_score(Y_true[col], Y_pred[:, n]))
        metrics_dict['accuracy'].append(accuracy_score(Y_true[col], Y_pred[:, n]))
        metrics_dict['category'].append(col)
    metrics_df = pd.DataFrame(data = metrics_dict)
    return metrics_df


def print_report(Y_true, Y_pred, category_names):
    """
    Print classification report for multioutput classification.
    """
    for n, col in enumerate(category_names):
        print("Report for {}".format(col))
        print(classification_report(Y_true[col], Y_pred[:, n]),'\n')

def evaluate_model(model, X_test, Y_test, category_names, database_filepath):
    """
    Predict on the test set and calculate classification metrics.
    """
    Y_pred = model.predict(X_test)
    print_report(Y_test, Y_pred, category_names)
    print('Saving classification metrics...')
    metrics_df = classification_metrics(Y_test, Y_pred, category_names)
    save_metrics(metrics_df, database_filepath)
    print('classification metrics saved')

def save_metrics(metrics_df, database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    metrics_df.to_sql('classification_metrics', engine, if_exists = 'replace')

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def load_model(model_filepath):
    model = pickle.load(open(model_filepath, 'rb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, database_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()