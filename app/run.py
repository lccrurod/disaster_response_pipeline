import json
import plotly
import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

class CustomUnpickler(pickle.Unpickler):
  """
  Unpickler object to specify modules used in the model construction.
  """
    def find_class(self, module, name):
        if name == 'tokenize':
            from extra_func import tokenize
            return tokenize
        if name == 'VerbCountExtractor':
            from extra_func import VerbCountExtractor
            return VerbCountExtractor
        return super().find_class(module, name)


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/messages_application.db')
df = pd.read_sql_table('messages_treated', engine)
metrics = pd.read_sql_table('classification_metrics', engine)

# load model
model = CustomUnpickler(open('../models/classifier.pkl', 'rb')).load()


# get data 

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # visual 1 data
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(gen_count*100/gen_count.sum(), 2)
    gen = list(gen_count.index)

    # visual 2 data
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    cat_num = cat_num.sort_values(ascending = False)
    cat = list(cat_num.index)

    colors = ['yellow', 'green', 'red']

    # visual 3 data
    metrics.sort_values(by = 'f1_score', ascending = False, inplace = True)
        
    # create visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.6,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": gen_per,
                  "y": gen
                },
                "marker": {
                  "colors": [
                    "#9d9d9c",
                    "#c00000",
                    "#007179"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": gen,
                "values": gen_count
              }
            ],
            "layout": {
              "title": "Count and Percent of Messages by Genre"
            }
        },
        {
            "data": [
              {
                "type": "bar",
                "y": cat_num,
                "x": cat,
                "marker": {
                  "color": 'c00000'}
                }
            ],
            "layout": {
              "title": "Count of Messages by Category",
              'yaxis': {
                  'title': "Count"
              },
              'barmode': 'group'
            }
        },
        {
            "data": [
              {
                "type": "bar",
                "y": metrics['f1_score'],
                "x": metrics['category'],
                "name": 'f1_score',
                "marker": {
                  "color": 'c00000'}
                },
              {
                "type": "bar",
                "y": metrics['accuracy'],
                "x": metrics['category'],
                "name": 'accuracy',
                "marker": {
                  "color": '9d9d9c'}
                }
            ],
            "layout": {
              "title": "Classification Performance by Category",
              'yaxis': {
                  'title': "Performance"
              },
              'barmode': 'group'
            }
        }
    ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()