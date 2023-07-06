import sys
import pandas as pd
import pickle
import joblib
from sqlalchemy import create_engine, inspect, text
import re

from custom_token_function import tokenize

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Load data from database into dataframe df, and split into features and target for model
    '''
    # load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    with engine.connect() as conn:
        df= conn.execute(text("SELECT * FROM DisasterTable")).fetchall()
        inspector = inspect(engine)
        columns = inspector.get_columns('DisasterTable')
        column_names = [column['name'] for column in columns]
        
    # Create dataframe from loaded data
    df = pd.DataFrame(df, columns= column_names)
    
    # Category names, also target variales
    category_names = ['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']
    
    # Set target variables and feature names
    X = df['message']
    Y = df[category_names]
    
    return X, Y, category_names



def build_model():
    '''
    Build model pipeline and split data into training and test dataset
    '''
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('classifier', MultiOutputClassifier(RandomForestClassifier()))])
    
    # Grid search to optimize hyperparameters
    parameters = parameters = {'classifier__estimator__n_estimators': [5],
                               'classifier__estimator__max_depth': [5],}

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model by generating a classification report for each category
    '''
    y_pred = model.predict(X_test)
    output = {}
    for i in range(len(category_names)):
        report= classification_report(y_pred[i], Y_test.iloc[i])
        output[category_names[i]] =  report
    print(output)


def save_model(model, model_filepath):
    '''
    Save model in pickle file for later use
    '''
    # save the model as a pickle file
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Generate model
    '''
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
        evaluate_model(model, X_test, Y_test, category_names)

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