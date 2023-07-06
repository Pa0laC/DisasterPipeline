# Disaster Response Pipeline Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Instruction](#Instructions)
3. [Results](#results)

## Project Motivation: <a name="motivation"></a>
The aim of this project is to help categorize the large quantity of messages left during a disaster response. To do this, we build an ETL and ML pipeline that will predict the themes of the text messages.

## Instructions: <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/

## Results: <a name="results"></a>
On the final webpage, two visualizations have been produced to consider the messages' distribution of genres, and the top 5 categories of messages we can find.
We do note that our training dataset was very imbalanced which might lead to an under-representation of the minority classes in our model. This might explain why some messages are incorrectly classified.