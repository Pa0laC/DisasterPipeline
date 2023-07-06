import json
import plotly
import pandas as pd
import joblib
import sys
sys.path.append('../models')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from custom_token_function import tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import dump, load
from sqlalchemy import create_engine


app = Flask(__name__)

#app.run(debug=True, port=3000)
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # first visual
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #second visualisation
    df_categories= df.drop(columns=['id','message', 'original', 'genre',])
    top_5_count = df_categories.sum().sort_values(ascending=False)[:5]
    top_5_name = top_5_count.index
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=top_5_name,
                    y=top_5_count
                )
            ],

            'layout': {
                'title': 'Top 5 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()